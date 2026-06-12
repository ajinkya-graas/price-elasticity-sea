#!/usr/bin/env python3
"""
Price Elasticity Analysis — Interactive HTML Report  v3
Generates one HTML per country  +  an index landing page.

Usage:
    python elasticity_report_v3.py
    python elasticity_report_v3.py --input  /path/to/sales.csv
                                   --meta   /path/to/mapping.xlsx
                                   --outdir /path/to/output/
    python elasticity_report_v3.py --top 500     # override SKU limit
"""

import argparse
import calendar
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
D_DAYS = {(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(11,11),(12,12)}
MIN_POINTS_FOR_ELASTICITY = 5
TOP_SKUS_PER_COUNTRY      = 500
DEFAULT_SELECTED_SKUS     = 5

COUNTRY_NAMES = {
    "PHP": "Philippines", "IDR": "Indonesia", "MYR": "Malaysia",
    "SGD": "Singapore",   "THB": "Thailand",  "VND": "Vietnam",
}
# ──────────────────────────────────────────────────────────────────────────────


def classify_day_type(dates: pd.Series) -> pd.Series:
    month = dates.dt.month
    day   = dates.dt.day
    is_dday = pd.Series(False, index=dates.index)
    for m, d in D_DAYS:
        is_dday |= (month == m) & (day == d)
    is_midmonth   = day.isin([14, 15])
    days_in_month = dates.apply(lambda dt: calendar.monthrange(dt.year, dt.month)[1])
    is_payday     = day > (days_in_month - 4)
    result        = pd.Series("BAU", index=dates.index)
    result[(is_midmonth | is_payday) & ~is_dday] = "Special"
    result[is_dday] = "D-day"
    return result


def channel_label(ch: str, country_channels: list) -> str:
    parts    = ch.split('-', 1)
    platform = parts[0].capitalize()
    same     = [c for c in country_channels if c.split('-')[0] == parts[0]]
    if len(same) > 1 and len(parts) > 1:
        return f"{platform} ({parts[1]})"
    return platform


def load_metadata(meta_path: str) -> pd.DataFrame:
    """Load and deduplicate the EAN → attribute mapping file."""
    print(f"Reading metadata {meta_path} …")
    mf = pd.read_excel(meta_path)
    mf.columns = [c.strip() for c in mf.columns]
    # Normalise column names
    rename = {}
    for col in mf.columns:
        lc = col.lower().replace(' ', '_')
        if 'color' in lc or 'colour' in lc:
            rename[col] = 'color_no'
        elif 'ean' in lc:
            rename[col] = 'ean'
        elif 'gender' in lc:
            rename[col] = 'gender'
        elif 'division' in lc:
            rename[col] = 'division'
        elif 'rbu' in lc:
            rename[col] = 'rbu'
    mf = mf.rename(columns=rename)
    # Strip float suffix (.0) that Excel adds when reading numeric EANs
    mf['ean'] = (mf['ean'].astype(str).str.strip()
                 .str.replace(r'\.0$', '', regex=True))
    for col in ['color_no', 'gender', 'division', 'rbu']:
        if col in mf.columns:
            mf[col] = mf[col].fillna('Unknown').astype(str).str.strip()
        else:
            mf[col] = 'Unknown'
    # One row per EAN (dedup — attributes are already 1:1)
    mf = mf.drop_duplicates(subset='ean')[['ean', 'color_no', 'gender', 'division', 'rbu']]
    print(f"  Metadata rows: {len(mf):,}  unique EANs: {mf['ean'].nunique():,}")
    return mf


def process_all(sales_path: str, meta_path: str, top_n: int) -> dict:
    print(f"\nReading sales data {sales_path} …")
    df = pd.read_csv(sales_path, low_memory=False)
    df.columns = ["date", "sku", "seller_id", "channel", "price", "qty",
                  "local_ccy", "product_name"]
    df["date"]         = pd.to_datetime(df["date"],  errors="coerce")
    df["price"]        = pd.to_numeric(df["price"],  errors="coerce")
    df["qty"]          = pd.to_numeric(df["qty"],    errors="coerce")
    df["sku"]          = df["sku"].astype(str).str.strip()
    df["channel"]      = df["channel"].astype(str).str.strip()
    df["local_ccy"]    = df["local_ccy"].astype(str).str.strip()
    df["product_name"] = df["product_name"].fillna("").astype(str).str.strip()
    df = df.dropna(subset=["date", "price", "qty", "local_ccy"])
    df = df[df["qty"] > 0]
    print(f"  Sales rows after cleaning: {len(df):,}")

    # Join metadata — colour_no is the analysis unit (SKU = colour + size)
    mf = load_metadata(meta_path)
    df = df.merge(mf, left_on="sku", right_on="ean", how="left")
    for col in ["color_no", "gender", "division", "rbu"]:
        df[col] = df[col].fillna("Unknown")
    # Unmapped SKUs: fall back to SKU itself as the colour_no so no data is lost
    df.loc[df["color_no"] == "Unknown", "color_no"] = df.loc[df["color_no"] == "Unknown", "sku"]
    matched_pct = (~df["color_no"].eq(df["sku"])).mean() * 100
    print(f"  Metadata match: {matched_pct:.1f}% of rows")

    df["day_type"] = classify_day_type(df["date"])
    # Normalise channel names: strip trailing "-<number>" so shopee-1 and shopee-6 merge
    df["channel"] = df["channel"].str.replace(r'-\d+$', '', regex=True)

    # ── Aggregate at COLOUR_NO level (folds in all sizes) ─────────────────────
    grp = ["color_no", "local_ccy", "channel", "date", "day_type"]

    agg_all = df.groupby(grp, as_index=False).agg(total_qty=("qty", "sum"))

    df_paid = df[df["price"] > 0].copy()
    df_paid["revenue"] = df_paid["price"] * df_paid["qty"]
    agg_paid = df_paid.groupby(grp, as_index=False).agg(
        paid_qty      = ("qty",     "sum"),
        total_revenue = ("revenue", "sum"),
    )
    # Weighted average price across all sizes at this colour on this day/channel
    agg_paid["avg_price"] = (agg_paid["total_revenue"] / agg_paid["paid_qty"]).round(2)

    agg = agg_all.merge(
        agg_paid[grp + ["paid_qty", "avg_price"]], on=grp, how="left"
    )
    agg["paid_qty"]  = agg["paid_qty"].fillna(0).astype(int)
    agg["avg_price"] = agg["avg_price"].fillna(0)
    agg["date_str"]  = agg["date"].dt.strftime("%Y-%m-%d")

    # Representative product name per colour_no: highest-qty SKU name
    color_names = (
        df.groupby(["color_no", "local_ccy"])
          .apply(lambda g: g.loc[g["qty"].idxmax(), "product_name"])
          .reset_index(name="product_name")
    )

    # Colour-level attribute lookup (1:1 by construction after metadata join)
    color_attrs = (
        df[df["color_no"] != df["sku"]]   # only properly mapped rows
          .drop_duplicates(subset=["color_no"])
          .set_index("color_no")[["gender", "division", "rbu"]]
    )

    global_min = agg["date"].min().strftime("%Y-%m-%d")
    global_max = agg["date"].max().strftime("%Y-%m-%d")

    countries_data = {}
    for ccy in sorted(df["local_ccy"].unique()):
        sub = agg[agg["local_ccy"] == ccy]
        raw = df[df["local_ccy"] == ccy]

        # Top N colours by total volume
        color_vol  = sub.groupby("color_no")["total_qty"].sum().sort_values(ascending=False)
        top_colors = color_vol.head(top_n).index.tolist()

        # Build colour metadata list
        names_sub = color_names[color_names["local_ccy"] == ccy].set_index("color_no")["product_name"]
        meta = []
        for cn in top_colors:
            name = names_sub.get(cn, cn)
            if cn in color_attrs.index:
                attrs = color_attrs.loc[cn]
                gender   = str(attrs.get("gender",   "Unknown") if isinstance(attrs, dict) else getattr(attrs, "gender",   "Unknown"))
                division = str(attrs.get("division", "Unknown") if isinstance(attrs, dict) else getattr(attrs, "division", "Unknown"))
                rbu      = str(attrs.get("rbu",      "Unknown") if isinstance(attrs, dict) else getattr(attrs, "rbu",      "Unknown"))
            else:
                gender = division = rbu = "Unknown"
            meta.append({"cn": cn, "name": name or cn,
                          "gender": gender, "division": division, "rbu": rbu})

        ccy_channels = sorted(raw["channel"].unique().tolist())

        genders   = sorted(set(m["gender"]   for m in meta if m["gender"]   != "Unknown"))
        divisions = sorted(set(m["division"] for m in meta if m["division"] != "Unknown"))
        rbus      = sorted(set(m["rbu"]      for m in meta if m["rbu"]      != "Unknown"))
        if any(m["gender"]   == "Unknown" for m in meta): genders.append("Unknown")
        if any(m["division"] == "Unknown" for m in meta): divisions.append("Unknown")
        if any(m["rbu"]      == "Unknown" for m in meta): rbus.append("Unknown")

        sub_top = sub[sub["color_no"].isin(top_colors)]
        records = []
        for row in sub_top.itertuples(index=False):
            records.append({
                "cn": row.color_no,
                "ch": row.channel,
                "d":  row.date_str,
                "t":  row.day_type,
                "p":  row.avg_price,
                "tq": row.total_qty,
                "pq": row.paid_qty,
            })

        countries_data[ccy] = {
            "ccy":       ccy,
            "country":   COUNTRY_NAMES.get(ccy, ccy),
            "channels":  ccy_channels,
            "ch_labels": {ch: channel_label(ch, ccy_channels) for ch in ccy_channels},
            "color_meta": meta,
            "def_colors": top_colors[:DEFAULT_SELECTED_SKUS],
            "genders":   genders,
            "divisions": divisions,
            "rbus":      rbus,
            "records":   records,
            "min_date":  sub["date"].min().strftime("%Y-%m-%d"),
            "max_date":  sub["date"].max().strftime("%Y-%m-%d"),
            "n_colors":  len(top_colors),
            "n_records": len(records),
        }
        print(f"  {ccy}: {len(top_colors)} colours, {len(records):,} records")

    return {"countries": countries_data, "min_date": global_min, "max_date": global_max}


# ── Country HTML template ──────────────────────────────────────────────────────
COUNTRY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Price Elasticity — __CCY__</title>
<script>__PLOTLY_JS__</script>
<style>
:root{--bg:#f8f9fa;--card:#fff;--border:#dee2e6;--primary:#2563eb;
  --dday:#ef4444;--special:#f59e0b;--bau:#3b82f6;--text:#1f2937;--muted:#6b7280}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text)}

.hdr{background:var(--card);border-bottom:1px solid var(--border);padding:12px 24px;display:flex;align-items:center;justify-content:space-between}
.hdr-left{display:flex;align-items:center;gap:14px}
.back-btn{padding:5px 12px;border:1px solid var(--border);border-radius:6px;background:#fff;font-size:12px;cursor:pointer;color:var(--muted);text-decoration:none;display:inline-flex;align-items:center;gap:4px}
.back-btn:hover{border-color:var(--primary);color:var(--primary)}
.hdr h1{font-size:17px;font-weight:600}
.hdr-sub{color:var(--muted);font-size:12px;margin-top:2px}
.ccy-badge{padding:3px 10px;background:#eff6ff;border:1px solid #bfdbfe;border-radius:20px;font-size:13px;font-weight:700;color:var(--primary)}

.filters{background:var(--card);border-bottom:1px solid var(--border);padding:10px 24px;display:flex;gap:18px;align-items:flex-end;flex-wrap:wrap}
.fg{display:flex;flex-direction:column;gap:4px}
.fl{font-size:10px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}

.toggle-grp{display:flex;gap:4px;flex-wrap:wrap}
.tgl{padding:4px 10px;border:1px solid var(--border);border-radius:5px;font-size:12px;font-weight:500;cursor:pointer;background:#fff;color:var(--muted);transition:all .12s}
.tgl.on{background:#eff6ff;border-color:var(--primary);color:var(--primary)}
.tgl-dday.on{background:#fef2f2;border-color:var(--dday);color:var(--dday)}
.tgl-special.on{background:#fffbeb;border-color:var(--special);color:#d97706}
.tgl-bau.on{background:#eff6ff;border-color:var(--bau);color:var(--bau)}

/* RBU dropdown */
.rbu-wrap{position:relative}
.rbu-btn{padding:5px 10px;border:1px solid var(--border);border-radius:6px;background:#fff;cursor:pointer;font-size:12px;min-width:160px;display:flex;justify-content:space-between;align-items:center;gap:6px}
.rbu-btn:hover{border-color:#94a3b8}
.rbu-drop{display:none;position:absolute;top:calc(100% + 4px);left:0;z-index:200;background:#fff;border:1px solid var(--border);border-radius:8px;box-shadow:0 6px 20px rgba(0,0,0,.12);width:220px;flex-direction:column;max-height:300px;overflow-y:auto}
.rbu-drop.open{display:flex;flex-direction:column}
.rbu-item{display:flex;align-items:center;gap:8px;padding:6px 12px;cursor:pointer;font-size:12px}
.rbu-item:hover{background:var(--bg)}
.rbu-item input{accent-color:var(--primary);cursor:pointer}
.rbu-actions{display:flex;gap:6px;padding:6px 8px;border-bottom:1px solid var(--border);flex-shrink:0}
.rbu-actions button{font-size:11px;padding:2px 8px;border:1px solid var(--border);border-radius:4px;cursor:pointer;background:#fff}

/* SKU dropdown */
.sku-wrap{position:relative}
.sku-btn{padding:6px 10px;border:1px solid var(--border);border-radius:6px;background:#fff;cursor:pointer;font-size:13px;min-width:230px;display:flex;justify-content:space-between;align-items:center;gap:8px}
.sku-btn:hover{border-color:#94a3b8}
.sku-drop{display:none;position:absolute;top:calc(100% + 4px);left:0;z-index:200;background:#fff;border:1px solid var(--border);border-radius:8px;box-shadow:0 6px 20px rgba(0,0,0,.12);width:360px;flex-direction:column}
.sku-drop.open{display:flex}
.sku-search{padding:8px;border-bottom:1px solid var(--border)}
.sku-search input{width:100%;padding:5px 8px;border:1px solid var(--border);border-radius:5px;font-size:13px;outline:none}
.sku-search input:focus{border-color:var(--primary)}
.sku-actions{display:flex;gap:6px;padding:6px 8px;border-bottom:1px solid var(--border)}
.sku-actions button{font-size:11px;padding:2px 8px;border:1px solid var(--border);border-radius:4px;cursor:pointer;background:#fff}
.sku-list{overflow-y:auto;max-height:270px;padding:4px 0}
.sku-item{display:flex;align-items:flex-start;gap:8px;padding:5px 12px;cursor:pointer;font-size:12px}
.sku-item:hover{background:var(--bg)}
.sku-item input{cursor:pointer;accent-color:var(--primary);flex-shrink:0;margin-top:3px}
.sku-name{font-weight:500;color:var(--text);line-height:1.3}
.sku-sub{color:var(--muted);font-size:10px;font-family:monospace;margin-top:1px}
.sku-tags{display:flex;gap:4px;margin-top:2px;flex-wrap:wrap}
.sku-tag{padding:1px 5px;border-radius:3px;font-size:9px;font-weight:600;background:#f3f4f6;color:#6b7280}

.dr{display:flex;gap:6px;align-items:center}
.dr input{padding:5px 8px;border:1px solid var(--border);border-radius:6px;font-size:13px}
.dr span{color:var(--muted);font-size:14px}

.section-desc{font-size:12px;color:var(--muted);padding:7px 24px;background:var(--card);border-bottom:1px solid var(--border);line-height:1.6}
.section-desc strong{color:#374151}

.stats{padding:12px 24px;display:flex;gap:12px}
.stat{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 16px;flex:1}
.stat-lbl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.stat-val{font-size:22px;font-weight:600;margin-top:2px;color:var(--text)}
.stat-sub{font-size:11px;color:var(--muted);margin-top:3px;line-height:1.5}
.stat-sub strong{color:#374151}

.grid{padding:0 24px 28px;display:grid;grid-template-columns:1fr 1fr;gap:14px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.card-title{font-size:13px;font-weight:600;padding:11px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px}
.card-hint{font-weight:400;color:var(--muted);font-size:10px}
.card-body{padding:4px}
.full{grid-column:1/-1}

.tbl-wrap{overflow-y:auto;max-height:400px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{padding:8px 12px;text-align:left;font-weight:600;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border);cursor:pointer;user-select:none;white-space:nowrap}
th:hover{color:var(--primary)}
td{padding:7px 12px;border-bottom:1px solid #f3f4f6;vertical-align:middle}
tr:hover td{background:var(--bg)}
.badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:10px;font-weight:600}
.b-el{background:#fef2f2;color:#dc2626}.b-in{background:#f0fdf4;color:#16a34a}
.b-un{background:#fffbeb;color:#d97706}.b-pos{background:#f5f3ff;color:#7c3aed}
.no-data{padding:36px 16px;text-align:center;color:var(--muted);font-size:13px}
#ch-scatter{min-height:360px}#ch-ts{min-height:280px}#ch-dt{min-height:260px}
.info{display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;border-radius:50%;background:#e5e7eb;color:#6b7280;font-size:10px;font-weight:700;cursor:default;position:relative;margin-left:4px;flex-shrink:0}
.info::after{content:attr(data-tip);display:none;position:absolute;top:calc(100%+6px);left:50%;transform:translateX(-50%);background:#1f2937;color:#fff;font-size:11px;font-weight:400;line-height:1.5;padding:8px 10px;border-radius:6px;width:260px;white-space:normal;z-index:999;box-shadow:0 4px 12px rgba(0,0,0,.2)}
.info:hover::after{display:block}
.divider{height:1px;background:var(--border);margin:0}
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-left">
    <a href="elasticity_index.html" class="back-btn">← All Markets</a>
    <div>
      <div style="display:flex;align-items:center;gap:10px">
        <img src="data:image/webp;base64,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" alt="PUMA" style="height:30px;width:auto">
        <h1>Price Elasticity Dashboard</h1>
        <span class="ccy-badge">__CCY__</span>
        <span style="color:var(--muted);font-size:13px">__COUNTRY__</span>
      </div>
      <div class="hdr-sub" id="data-summary">Loading…</div>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;align-items:flex-end;gap:2px">
    <div style="font-size:9px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase">Powered by</div>
    <img src="data:image/png;base64,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" alt="Graas" style="height:28px;width:auto">
  </div>
</div>

<!-- Filter row 1: Channel + Metadata filters -->
<div class="filters">
  <div class="fg">
    <div class="fl">Channel</div>
    <div class="toggle-grp" id="ch-filter"></div>
  </div>
  <div class="fg">
    <div class="fl">Gender</div>
    <div class="toggle-grp" id="gender-filter"></div>
  </div>
  <div class="fg">
    <div class="fl">Product Division</div>
    <div class="toggle-grp" id="division-filter"></div>
  </div>
  <div class="fg">
    <div class="fl">RBU</div>
    <div class="rbu-wrap">
      <button class="rbu-btn" onclick="toggleRbuDrop()" id="rbu-btn">
        <span id="rbu-lbl">All</span><span>▾</span>
      </button>
      <div class="rbu-drop" id="rbu-drop">
        <div class="rbu-actions">
          <button onclick="rbuAll()">All</button>
          <button onclick="rbuClear()">Clear</button>
        </div>
        <div id="rbu-list"></div>
      </div>
    </div>
  </div>
</div>

<!-- Filter row 2: Colour + Day type + Date -->
<div class="filters">
  <div class="fg">
    <div class="fl">Colour / Product <span style="color:var(--muted);font-weight:400;text-transform:none;letter-spacing:0">— filtered by above</span></div>
    <div class="sku-wrap">
      <button class="sku-btn" onclick="toggleSkuDrop()" id="sku-btn">
        <span id="sku-lbl">Loading…</span><span>▾</span>
      </button>
      <div class="sku-drop" id="sku-drop">
        <div class="sku-search">
          <input id="sku-q" type="text" placeholder="Search by name or colour code…" oninput="renderSkuList(this.value)">
        </div>
        <div class="sku-actions">
          <button onclick="selectAll()">All visible</button>
          <button onclick="clearAll()">Clear</button>
          <button onclick="resetDefault()">Top 5</button>
        </div>
        <div class="sku-list" id="sku-list"></div>
      </div>
    </div>
  </div>

  <div class="fg">
    <div class="fl">Day Type <span class="info" data-tip="D-day = mega sale campaigns (3.3–12.12). Special = mid-month (14–15) &amp; payday (last 4 days). BAU = everything else.">i</span></div>
    <div class="toggle-grp">
      <button class="tgl tgl-dday on"    id="t-dday"    onclick="toggleDT('D-day')">D-day</button>
      <button class="tgl tgl-special on" id="t-special" onclick="toggleDT('Special')">Special</button>
      <button class="tgl tgl-bau on"     id="t-bau"     onclick="toggleDT('BAU')">BAU</button>
    </div>
  </div>

  <div class="fg">
    <div class="fl">Date Range</div>
    <div class="dr">
      <input type="date" id="d0" onchange="onDate()">
      <span>→</span>
      <input type="date" id="d1" onchange="onDate()">
    </div>
  </div>
</div>

<div class="section-desc">
  Top __N_COLORS__ colours by volume · Each colour aggregates all sizes · <strong>D-day</strong> = mega sale campaigns (3.3–12.12) ·
  <strong>Special</strong> = mid-month (14–15) &amp; payday (last 4 days) · <strong>BAU</strong> = everything else.
  Gender / Division / RBU filters narrow the colour list above.
</div>

<div class="stats">
  <div class="stat"><div class="stat-lbl">SKUs w/ Elasticity</div><div class="stat-val" id="s-skus">—</div><div class="stat-sub">SKUs with ≥5 unique price points in current filter view</div></div>
  <div class="stat"><div class="stat-lbl">Avg Elasticity</div><div class="stat-val" id="s-el">—</div><div class="stat-sub"><strong>Below −1</strong> = elastic · <strong>−1 to 0</strong> = inelastic · <strong>Positive</strong> = anomalous</div></div>
  <div class="stat"><div class="stat-lbl">Total Units Sold</div><div class="stat-val" id="s-qty">—</div><div class="stat-sub">Sum across selected SKUs, channels &amp; date range</div></div>
  <div class="stat"><div class="stat-lbl">Avg Price</div><div class="stat-val" id="s-price">—</div><div class="stat-sub">Weighted average selling price (weighted by qty sold)</div></div>
</div>

<div class="grid">
  <div class="card">
    <div class="card-title">Price vs. Quantity
      <span class="info" data-tip="Each dot = one unique price point for a SKU. The dotted line is the estimated demand curve (log-log OLS). Downward slope = higher price → lower demand.">i</span>
    </div>
    <div class="card-body"><div id="ch-scatter"></div></div>
  </div>
  <div class="card">
    <div class="card-title">Elasticity by Colour <span class="card-hint">click headers to sort</span>
      <span class="info" data-tip="Below −1 (Elastic): price-sensitive, demand drops sharply. −1 to 0 (Inelastic): demand holds. Positive: unusual — check data.">i</span>
      <button onclick="exportTableCSV()" style="margin-left:auto;padding:3px 10px;font-size:11px;font-weight:500;border:1px solid var(--border);border-radius:5px;background:#f9fafb;cursor:pointer;color:var(--text)" title="Download table as CSV">⬇ Export CSV</button>
    </div>
    <div class="card-body" id="tbl-wrap"></div>
  </div>
  <div class="card full">
    <div class="card-title">Price &amp; Quantity Over Time
      <span class="info" data-tip="Blue line = avg selling price (left axis). Bars = total units sold (right axis). Spot whether demand spikes are price-driven or organic.">i</span>
    </div>
    <div class="card-body"><div id="ch-ts"></div></div>
  </div>
  <div class="card full">
    <div class="card-title">Elasticity by Day Type
      <span class="info" data-tip="Compares price sensitivity across D-days, Special days, and BAU. More negative on D-days = customers are more price-driven during campaigns.">i</span>
    </div>
    <div class="card-body"><div id="ch-dt"></div></div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;
const DATA_CCY = '__CCY__';

// ── State ─────────────────────────────────────────────────────────────────────
let selChannels = new Set(DATA.channels);
let selGender   = new Set(DATA.genders);
let selDivision = new Set(DATA.divisions);
let selRBU      = new Set(DATA.rbus);
let sel         = new Set(DATA.def_colors.map(String));
let selDT       = new Set(['D-day','Special','BAU']);
let d0          = DATA.min_date;
let d1          = DATA.max_date;
let sortCol     = 'total_qty';
let sortAsc     = false;

// ── Plotly constants ──────────────────────────────────────────────────────────
const DC      = {'D-day':'#ef4444','Special':'#f59e0b','BAU':'#3b82f6'};
const PLTCFG  = {responsive:true,displayModeBar:false};
const PLTFONT = {family:"-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif",size:11};
const PLTBG   = {paper_bgcolor:'white',plot_bgcolor:'white'};
const GRID    = {showgrid:true,gridcolor:'#f3f4f6'};

// ── Colour lookup helpers ─────────────────────────────────────────────────────
const COLOR_MAP = {};
DATA.color_meta.forEach(m=>{ COLOR_MAP[m.cn]=m; });
function colorInfo(cn){ return COLOR_MAP[cn]||{cn,name:cn,gender:'',division:'',rbu:''}; }

// ── Visible colours (intersection of metadata filters) ────────────────────────
function visibleColors(){
  return DATA.color_meta
    .filter(m=>
      (selGender.has(m.gender)||selGender.has('Unknown'))&&
      (selDivision.has(m.division)||selDivision.has('Unknown'))&&
      (selRBU.has(m.rbu)||selRBU.has('Unknown'))
    )
    .map(m=>m.cn);
}
function visibleColorSet(){ return new Set(visibleColors()); }

// ── Init ──────────────────────────────────────────────────────────────────────
function init(){
  const s0=document.getElementById('d0'), s1=document.getElementById('d1');
  [s0,s1].forEach(el=>{ el.min=DATA.min_date; el.max=DATA.max_date; });
  s0.value=DATA.min_date; s1.value=DATA.max_date;

  document.getElementById('data-summary').textContent=
    `${DATA.n_records.toLocaleString()} daily aggregates · top ${DATA.n_colors} colours · ${DATA.min_date} → ${DATA.max_date}`;

  renderChannelFilter();
  renderMetaFilter('gender-filter', DATA.genders, selGender, toggleGender);
  renderMetaFilter('division-filter', DATA.divisions, selDivision, toggleDivision);
  renderRbuList();
  updRbuBtn();
  renderSkuList('');

  // Close dropdowns on outside click
  document.addEventListener('click', e=>{
    if(!document.querySelector('.sku-wrap').contains(e.target))
      document.getElementById('sku-drop').classList.remove('open');
    if(!document.querySelector('.rbu-wrap').contains(e.target))
      document.getElementById('rbu-drop').classList.remove('open');
  });

  updateAll();
}

// ── Channel filter ────────────────────────────────────────────────────────────
function renderChannelFilter(){
  const wrap=document.getElementById('ch-filter');
  wrap.innerHTML=DATA.channels.map(ch=>`
    <button class="tgl ${selChannels.has(ch)?'on':''}"
            data-ch="${ch}" onclick="toggleCh('${ch}')">${DATA.ch_labels[ch]||ch}</button>
  `).join('');
}
function toggleCh(ch){
  if(selChannels.has(ch)){ if(selChannels.size===1)return; selChannels.delete(ch); }
  else selChannels.add(ch);
  document.querySelector(`.tgl[data-ch="${ch}"]`).classList.toggle('on',selChannels.has(ch));
  updateAll();
}

// ── Generic metadata toggle filter ───────────────────────────────────────────
function renderMetaFilter(elId, vals, stateSet, toggleFn){
  document.getElementById(elId).innerHTML=vals.map(v=>`
    <button class="tgl ${stateSet.has(v)?'on':''}"
            data-val="${v}" onclick="${toggleFn.name}('${v}')">${v}</button>
  `).join('');
}
function toggleMeta(val, stateSet, elId, refreshFn){
  if(stateSet.has(val)){ if(stateSet.size===1)return; stateSet.delete(val); }
  else stateSet.add(val);
  document.querySelector(`#${elId} .tgl[data-val="${val}"]`).classList.toggle('on',stateSet.has(val));
  refreshFn();
  renderSkuList(document.getElementById('sku-q').value);
  updateAll();
}
function toggleGender(v)  { toggleMeta(v,selGender,  'gender-filter',  ()=>renderMetaFilter('gender-filter',  DATA.genders,   selGender,   toggleGender));   }
function toggleDivision(v){ toggleMeta(v,selDivision,'division-filter',()=>renderMetaFilter('division-filter',DATA.divisions, selDivision, toggleDivision)); }

// ── RBU dropdown ──────────────────────────────────────────────────────────────
function renderRbuList(){
  document.getElementById('rbu-list').innerHTML=DATA.rbus.map(r=>`
    <label class="rbu-item">
      <input type="checkbox" value="${r}" ${selRBU.has(r)?'checked':''} onchange="toggleRbu('${r}',this.checked)">
      ${r}
    </label>`).join('');
}
function toggleRbuDrop(){ document.getElementById('rbu-drop').classList.toggle('open'); }
function toggleRbu(r,c){
  if(!c && selRBU.size===1){ document.querySelector(`#rbu-list input[value="${r}"]`).checked=true; return; }
  c?selRBU.add(r):selRBU.delete(r);
  updRbuBtn();
  renderSkuList(document.getElementById('sku-q').value);
  updateAll();
}
function rbuAll(){ DATA.rbus.forEach(r=>selRBU.add(r)); renderRbuList(); updRbuBtn(); renderSkuList(document.getElementById('sku-q').value); updateAll(); }
function rbuClear(){
  const first=DATA.rbus[0]; selRBU.clear(); if(first)selRBU.add(first);
  renderRbuList(); updRbuBtn(); renderSkuList(document.getElementById('sku-q').value); updateAll();
}
function updRbuBtn(){
  document.getElementById('rbu-lbl').textContent=
    selRBU.size===DATA.rbus.length?'All':
    selRBU.size===1?[...selRBU][0]:
    `${selRBU.size} selected`;
}

// ── Colour multi-select ───────────────────────────────────────────────────────
function renderSkuList(q){
  const lq=q.toLowerCase();
  const vis=visibleColorSet();
  const items=DATA.color_meta.filter(m=>
    vis.has(m.cn) &&
    (!q || m.cn.toLowerCase().includes(lq) || m.name.toLowerCase().includes(lq))
  );
  document.getElementById('sku-list').innerHTML=items.length===0
    ?'<div style="padding:20px;text-align:center;color:var(--muted);font-size:12px">No colours match current filters</div>'
    :items.map(m=>`
      <label class="sku-item">
        <input type="checkbox" value="${m.cn}" ${sel.has(m.cn)?'checked':''} onchange="onSku('${m.cn}',this.checked)">
        <div>
          <div class="sku-name">${m.name||m.cn}</div>
          <div class="sku-sub">${m.cn}</div>
          <div class="sku-tags">
            ${m.gender&&m.gender!=='Unknown'?`<span class="sku-tag">${m.gender}</span>`:''}
            ${m.division&&m.division!=='Unknown'?`<span class="sku-tag">${m.division}</span>`:''}
            ${m.rbu&&m.rbu!=='Unknown'?`<span class="sku-tag">${m.rbu}</span>`:''}
          </div>
        </div>
      </label>`).join('');
  updSkuBtn();
}
function toggleSkuDrop(){ document.getElementById('sku-drop').classList.toggle('open'); }
function onSku(cn,c){ c?sel.add(cn):sel.delete(cn); updSkuBtn(); updateAll(); }
function selectAll(){
  const q=document.getElementById('sku-q').value.toLowerCase();
  const vis=visibleColorSet();
  DATA.color_meta.filter(m=>vis.has(m.cn)&&(!q||m.cn.toLowerCase().includes(q)||m.name.toLowerCase().includes(q))).forEach(m=>sel.add(m.cn));
  renderSkuList(document.getElementById('sku-q').value); updateAll();
}
function clearAll(){  sel.clear(); renderSkuList(document.getElementById('sku-q').value); updateAll(); }
function resetDefault(){ sel=new Set(DATA.def_colors.map(String)); renderSkuList(document.getElementById('sku-q').value); updateAll(); }
function updSkuBtn(){
  const active=[...sel].filter(cn=>visibleColorSet().has(cn));
  const lbl=active.length===0?'None selected'
    :active.length===1?(colorInfo(active[0]).name||active[0])
    :`${active.length} colours selected`;
  const el=document.getElementById('sku-lbl');
  el.textContent=lbl.length>35?lbl.slice(0,33)+'…':lbl;
}

// ── Day-type toggles ──────────────────────────────────────────────────────────
const DT_CLS={'D-day':'tgl-dday','Special':'tgl-special','BAU':'tgl-bau'};
const DT_ID ={'D-day':'t-dday','Special':'t-special','BAU':'t-bau'};
function toggleDT(dt){
  if(selDT.has(dt)){ if(selDT.size===1)return; selDT.delete(dt); }
  else selDT.add(dt);
  document.getElementById(DT_ID[dt]).className=
    'tgl '+DT_CLS[dt]+(selDT.has(dt)?' on':'');
  updateAll();
}

// ── Date filter ───────────────────────────────────────────────────────────────
function onDate(){
  d0=document.getElementById('d0').value;
  d1=document.getElementById('d1').value;
  updateAll();
}

// ── Main filter ───────────────────────────────────────────────────────────────
function filtered(){
  const vis=visibleColorSet();
  return DATA.records.filter(r=>
    selChannels.has(r.ch) &&
    sel.has(r.cn) &&
    vis.has(r.cn) &&
    selDT.has(r.t) &&
    r.d>=d0 && r.d<=d1
  );
}

// ── OLS elasticity ────────────────────────────────────────────────────────────
function ols(prices,qtys){
  const n=prices.length; if(n<5)return null;
  const lp=prices.map(p=>Math.log(p)),lq=qtys.map(q=>Math.log(q));
  const mp=lp.reduce((a,b)=>a+b,0)/n,mq=lq.reduce((a,b)=>a+b,0)/n;
  const cov=lp.reduce((s,p,i)=>s+(p-mp)*(lq[i]-mq),0);
  const vp=lp.reduce((s,p)=>s+(p-mp)**2,0);
  return vp>0?cov/vp:null;
}

function elByColor(recs){
  const m={};
  recs.forEach(r=>{
    if(!m[r.cn])m[r.cn]={p:[],q:[],totalQty:0,paidQty:0,rev:0};
    m[r.cn].totalQty+=r.tq;
    if(r.pq>0){
      m[r.cn].p.push(r.p);m[r.cn].q.push(r.pq);
      m[r.cn].paidQty+=r.pq;m[r.cn].rev+=r.p*r.pq;
    }
  });
  return Object.entries(m).map(([cn,d])=>{
    const info=colorInfo(cn);
    return {
      cn, name:info.name||cn,
      gender:info.gender||'', division:info.division||'', rbu:info.rbu||'',
      elasticity:ols(d.p,d.q),
      avg_price:d.paidQty>0?d.rev/d.paidQty:0,
      total_qty:d.totalQty, data_points:d.p.length
    };
  }).filter(r=>r.elasticity!==null);
}

// ── Plotly wrapper ────────────────────────────────────────────────────────────
function pReact(id,traces,layout,cfg){
  try{ Plotly.react(id,traces,layout,cfg||PLTCFG); }
  catch(e){
    const el=document.getElementById(id);
    if(el&&!el.querySelector('.err'))
      el.innerHTML=`<div class="err" style="padding:40px;text-align:center;color:#9ca3af;font-size:13px">⚠️ ${e.message}</div>`;
  }
}

// ── Update all ────────────────────────────────────────────────────────────────
function updateAll(){
  const f=filtered(),el=elByColor(f);
  updStats(f,el); updTable(el); updScatter(f,el); updTS(f); updDT(f);
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function updStats(f,el){
  const qty=f.reduce((s,r)=>s+r.tq,0);
  const rev=f.reduce((s,r)=>s+r.p*r.tq,0);
  const avgEl=el.length?el.reduce((s,r)=>s+r.elasticity,0)/el.length:null;
  document.getElementById('s-skus').textContent=el.length;
  document.getElementById('s-el').textContent=avgEl!==null?avgEl.toFixed(2):'—';
  document.getElementById('s-qty').textContent=qty.toLocaleString();
  document.getElementById('s-price').textContent=qty>0?Math.round(rev/qty).toLocaleString():'—';
}

// ── Scatter ───────────────────────────────────────────────────────────────────
function updScatter(f,el){
  if(!f.length){ pReact('ch-scatter',[],{height:360,...PLTBG,font:PLTFONT,annotations:[{text:'No data',showarrow:false,xref:'paper',yref:'paper',x:.5,y:.5,font:{size:14,color:'#9ca3af'}}]}); return; }
  const byPriceKey={};
  f.forEach(r=>{
    if(r.pq===0)return;
    const p=Math.round(r.p),key=`${r.cn}||${p}`;
    if(!byPriceKey[key])byPriceKey[key]={cn:r.cn,price:p,qty:0,days:0};
    byPriceKey[key].qty+=r.pq; byPriceKey[key].days+=1;
  });
  const pts=Object.values(byPriceKey).filter(p=>p.qty>0);
  if(!pts.length){ pReact('ch-scatter',[],{height:360,...PLTBG,font:PLTFONT,annotations:[{text:'No data',showarrow:false,xref:'paper',yref:'paper',x:.5,y:.5,font:{size:14,color:'#9ca3af'}}]}); return; }
  const xs=[],ys=[],ts=[];
  pts.forEach(p=>{
    const nm=colorInfo(p.cn).name||p.cn; const short=nm.length>30?nm.slice(0,28)+'…':nm;
    xs.push(p.price); ys.push(p.qty);
    ts.push(`${short}<br>${p.cn}<br>Price: ${p.price.toLocaleString()}<br>Qty: ${p.qty.toLocaleString()}<br>Days: ${p.days}`);
  });
  const traces=[{type:'scatter',mode:'markers',name:'Data',showlegend:false,
    x:xs,y:ys,text:ts,hoverinfo:'text',marker:{color:'#3b82f6',opacity:.5,size:6}}];
  // OLS trendline in log-log space — straight line between min and max price
  const lps=pts.map(p=>Math.log(p.price)),lqs=pts.map(p=>Math.log(p.qty));
  const n=lps.length,slp=lps.reduce((a,b)=>a+b,0)/n,slq=lqs.reduce((a,b)=>a+b,0)/n;
  const num=lps.reduce((s,lp,i)=>s+(lp-slp)*(lqs[i]-slq),0);
  const den=lps.reduce((s,lp)=>s+(lp-slp)**2,0);
  if(den>0&&n>=5){
    const e=num/den,mn=Math.min(...pts.map(p=>p.price)),mx=Math.max(...pts.map(p=>p.price));
    traces.push({type:'scatter',mode:'lines',name:`Overall e=${e.toFixed(2)}`,
      x:[mn,mx],y:[Math.exp(slq+e*(Math.log(mn)-slp)),Math.exp(slq+e*(Math.log(mx)-slp))],
      line:{color:'#dc2626',width:2,dash:'dot'},
      hovertemplate:`Elasticity: ${e.toFixed(2)}<extra></extra>`});
  }
  pReact('ch-scatter',traces,{height:360,margin:{t:10,r:10,b:50,l:60},
    xaxis:{title:'Price',type:'log',...GRID},
    yaxis:{title:'Total Quantity Sold',type:'log',...GRID},
    legend:{orientation:'h',y:-0.22},...PLTBG,font:PLTFONT});
}

// ── Elasticity table ──────────────────────────────────────────────────────────
let lastTableData=[];
function exportTableCSV(){
  if(!lastTableData.length)return;
  const cols=['Colour No','Product','Division','RBU','Gender','Elasticity','Avg Price','Total Qty','Data Points'];
  const rows=lastTableData.map(r=>[
    r.cn, `"${(r.name||'').replace(/"/g,'""')}"`,
    r.division||'', r.rbu||'', r.gender||'',
    r.elasticity.toFixed(2), Math.round(r.avg_price), r.total_qty, r.data_points
  ].join(','));
  const csv=[cols.join(','),...rows].join('\n');
  const a=document.createElement('a');
  a.href='data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
  a.download=`elasticity_${DATA_CCY||'export'}.csv`;
  a.click();
}
function badgeCls(e){ return e<-1?'b-el':e<0?'b-in':e<0.1?'b-un':'b-pos'; }
function badgeTxt(e){ return e<-1?'Elastic':e<0?'Inelastic':e<0.1?'Unitary':'Positive?'; }
function updTable(el){
  const s=[...el].sort((a,b)=>{
    const av=a[sortCol]??-Infinity,bv=b[sortCol]??-Infinity;
    return sortAsc?(av>bv?1:-1):(av<bv?1:-1);
  });
  lastTableData=s;
  const si=c=>c===sortCol?(sortAsc?' ↑':' ↓'):'';
  const html=s.length===0
    ?'<div class="no-data">No SKUs with ≥5 data points in current selection</div>'
    :`<div class="tbl-wrap"><table>
      <thead><tr>
        <th onclick="sort('name')">Product${si('name')}</th>
        <th onclick="sort('cn')">Colour No${si('cn')}</th>
        <th onclick="sort('division')">Division${si('division')}</th>
        <th onclick="sort('rbu')">RBU${si('rbu')}</th>
        <th onclick="sort('gender')">Gender${si('gender')}</th>
        <th onclick="sort('elasticity')">Elasticity${si('elasticity')}</th>
        <th onclick="sort('avg_price')">Avg Price${si('avg_price')}</th>
        <th onclick="sort('total_qty')">Total Qty${si('total_qty')}</th>
        <th onclick="sort('data_points')">Pts${si('data_points')}</th>
      </tr></thead>
      <tbody>${s.map(r=>`<tr>
        <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${r.name}">${r.name}</td>
        <td style="font-family:monospace;font-size:10px;color:var(--muted)">${r.cn}</td>
        <td style="font-size:11px;color:var(--muted)">${r.division||'—'}</td>
        <td style="font-size:11px;color:var(--muted);max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${r.rbu||''}">${r.rbu||'—'}</td>
        <td style="font-size:11px;color:var(--muted)">${r.gender||'—'}</td>
        <td><span class="badge ${badgeCls(r.elasticity)}">${r.elasticity.toFixed(2)}</span>
            <span style="font-size:10px;color:#9ca3af;margin-left:4px">${badgeTxt(r.elasticity)}</span></td>
        <td>${Math.round(r.avg_price).toLocaleString()}</td>
        <td>${r.total_qty.toLocaleString()}</td>
        <td style="color:${r.data_points<10?'#f59e0b':'inherit'}">${r.data_points}</td>
      </tr>`).join('')}</tbody>
    </table></div>`;
  document.getElementById('tbl-wrap').innerHTML=html;
}
function sort(col){ sortAsc=sortCol===col?!sortAsc:false; sortCol=col; updateAll(); }

// ── Time series ───────────────────────────────────────────────────────────────
function updTS(f){
  if(!f.length){ pReact('ch-ts',[],{height:280,...PLTBG,font:PLTFONT}); return; }
  const byDate={};
  f.forEach(r=>{ if(!byDate[r.d])byDate[r.d]={qty:0,rev:0}; byDate[r.d].qty+=r.tq; byDate[r.d].rev+=r.p*r.tq; });
  const dates=Object.keys(byDate).sort();
  pReact('ch-ts',[
    {type:'bar',name:'Total Qty',x:dates,y:dates.map(d=>byDate[d].qty),
     marker:{color:'#bfdbfe'},yaxis:'y2',hovertemplate:'%{x}<br>Qty: %{y:,}<extra></extra>'},
    {type:'scatter',mode:'lines+markers',name:'Avg Price',x:dates,
     y:dates.map(d=>byDate[d].qty>0?byDate[d].rev/byDate[d].qty:null),
     line:{color:'#2563eb',width:2},marker:{size:3},hovertemplate:'%{x}<br>Price: %{y:.2f}<extra></extra>'},
  ],{height:280,margin:{t:10,r:64,b:50,l:60},xaxis:{showgrid:false},
    yaxis:{title:'Avg Price',...GRID},
    yaxis2:{title:'Total Qty',overlaying:'y',side:'right',showgrid:false},
    legend:{orientation:'h',y:-0.28},...PLTBG,font:PLTFONT},PLTCFG);
}

// ── Day type chart ────────────────────────────────────────────────────────────
function updDT(f){
  const order=['D-day','Special','BAU'],byDT={};
  f.forEach(r=>{
    if(!byDT[r.t])byDT[r.t]={};
    if(!byDT[r.t][r.cn])byDT[r.t][r.cn]={p:[],q:[]};
    byDT[r.t][r.cn].p.push(r.p); byDT[r.t][r.cn].q.push(r.tq);
  });
  const vals={};
  order.forEach(dt=>{
    if(!byDT[dt])return;
    const es=Object.values(byDT[dt]).map(d=>ols(d.p,d.q)).filter(e=>e!==null);
    if(es.length)vals[dt]=es.reduce((a,b)=>a+b,0)/es.length;
  });
  const dts=order.filter(dt=>vals[dt]!==undefined);
  if(!dts.length){ pReact('ch-dt',[],{height:260,...PLTBG,font:PLTFONT}); return; }
  pReact('ch-dt',[{type:'bar',x:dts,y:dts.map(dt=>vals[dt]),
    marker:{color:dts.map(dt=>DC[dt])},text:dts.map(dt=>vals[dt].toFixed(2)),
    textposition:'outside',hovertemplate:'%{x}<br>Avg Elasticity: %{y:.2f}<extra></extra>'}],
    {height:260,margin:{t:30,r:20,b:50,l:70},
     yaxis:{title:'Avg Elasticity',...GRID,zeroline:true,zerolinecolor:'#d1d5db'},
     xaxis:{showgrid:false},
     shapes:[{type:'line',x0:-.5,x1:dts.length-.5,y0:-1,y1:-1,line:{color:'#9ca3af',dash:'dot',width:1}}],
     annotations:[{x:dts.length-.5,y:-1,text:'Unitary (−1)',showarrow:false,xanchor:'right',font:{size:10,color:'#9ca3af'}}],
     ...PLTBG,font:PLTFONT},PLTCFG);
}

// ── Start ─────────────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>
"""

# ── Index page ─────────────────────────────────────────────────────────────────
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Price Elasticity Dashboard</title>
<style>
:root{--bg:#f0f4f8;--card:#fff;--border:#dee2e6;--primary:#2563eb;--text:#1f2937;--muted:#6b7280}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}

/* Header */
.hdr{background:#fff;border-bottom:2px solid #e5e7eb;padding:14px 32px}
.hdr-inner{display:flex;align-items:center;justify-content:space-between;max-width:1040px;margin:0 auto}
.hdr-brand{display:flex;align-items:center;gap:20px}
.hdr-divider{width:1px;height:36px;background:#e5e7eb}
.hdr-titles h1{font-size:17px;font-weight:700;color:#1f2937;letter-spacing:-.01em;margin:0}
.hdr-sub{font-size:11px;margin-top:3px;color:#6b7280}
.graas-side{display:flex;flex-direction:column;align-items:flex-end;gap:4px}
.graas-pill{background:#0f172a;border-radius:8px;padding:6px 12px;display:flex;flex-direction:column;align-items:center;gap:3px}
.powered-lbl{font-size:9px;color:rgba(255,255,255,.55);letter-spacing:.08em;text-transform:uppercase}
.graas-tagline{font-size:9px;color:#9ca3af;text-align:right}

/* Main layout */
.main{max-width:1040px;margin:0 auto;padding:24px 24px 40px}

/* Country grid — shown first, above fold */
.section-label{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:28px}
@media(max-width:680px){.grid{grid-template-columns:1fr 1fr}}
@media(max-width:420px){.grid{grid-template-columns:1fr}}

.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px 20px;text-decoration:none;color:inherit;display:block;transition:all .15s;position:relative;overflow:hidden}
.card:hover{border-color:var(--primary);box-shadow:0 4px 16px rgba(37,99,235,.12);transform:translateY(-2px)}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--primary);opacity:0;transition:opacity .15s}
.card:hover::before{opacity:1}
.card-top{display:flex;align-items:flex-start;justify-content:space-between}
.card-ccy{font-size:28px;font-weight:800;color:var(--primary);line-height:1}
.card-country{font-size:13px;font-weight:600;color:var(--text);margin-top:3px}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;background:#eff6ff;color:var(--primary)}
.card-stats{margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:6px 12px}
.card-stat-lbl{font-size:10px;color:var(--muted)}
.card-stat-val{font-size:13px;font-weight:600;color:var(--text)}
.card-date{font-size:10px;color:var(--muted);margin-top:10px;padding-top:10px;border-top:1px solid var(--border)}
.open-btn{margin-top:12px;display:flex;align-items:center;justify-content:center;gap:6px;padding:6px;border:1px solid var(--border);border-radius:6px;font-size:12px;font-weight:500;color:var(--primary);background:#f8faff;transition:all .12s}
.card:hover .open-btn{background:var(--primary);color:#fff;border-color:var(--primary)}

/* Explainer — collapsible, below the fold */
.explainer-toggle{display:flex;align-items:center;justify-content:space-between;cursor:pointer;padding:14px 18px;background:var(--card);border:1px solid var(--border);border-radius:10px;margin-bottom:0;user-select:none}
.explainer-toggle:hover{border-color:#94a3b8}
.explainer-toggle h2{font-size:14px;font-weight:600;color:var(--text)}
.explainer-toggle .arrow{font-size:13px;color:var(--muted);transition:transform .2s}
.explainer-toggle.open .arrow{transform:rotate(180deg)}
.explainer-body{background:var(--card);border:1px solid var(--border);border-top:none;border-radius:0 0 10px 10px;padding:20px 22px;display:none}
.explainer-body.open{display:block}

.explainer-body p{font-size:13px;color:#374151;line-height:1.7;margin-bottom:10px}
.explainer-body p:last-child{margin-bottom:0}
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}
@media(max-width:700px){.steps{grid-template-columns:1fr 1fr}}
.step{background:#f8faff;border:1px solid #dbeafe;border-radius:8px;padding:12px}
.step-num{font-size:10px;font-weight:700;color:var(--primary);text-transform:uppercase;letter-spacing:.06em}
.step-title{font-size:12px;font-weight:600;color:var(--text);margin:3px 0}
.step-desc{font-size:11px;color:var(--muted);line-height:1.5}
.interp{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}
@media(max-width:600px){.interp{grid-template-columns:1fr}}
.interp-card{border-radius:7px;padding:12px 14px}
.interp-card.elastic{background:#fef2f2;border:1px solid #fecaca}
.interp-card.inelastic{background:#f0fdf4;border:1px solid #bbf7d0}
.interp-card.positive{background:#fffbeb;border:1px solid #fde68a}
.interp-title{font-size:12px;font-weight:700;margin-bottom:4px}
.interp-card.elastic .interp-title{color:#dc2626}
.interp-card.inelastic .interp-title{color:#16a34a}
.interp-card.positive .interp-title{color:#d97706}
.interp-desc{font-size:11px;color:#374151;line-height:1.5}
.note{font-size:12px;color:var(--muted);background:#f9fafb;border-left:3px solid var(--border);padding:8px 12px;border-radius:0 5px 5px 0;margin-top:12px;line-height:1.6}

</style>
</head>
<body>
<div class="hdr">
  <div class="hdr-inner">
    <div class="hdr-brand">
      <img src="data:image/webp;base64,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" alt="PUMA" style="height:52px;width:auto">
      <div class="hdr-divider"></div>
      <div class="hdr-titles">
        <h1>Price Elasticity Dashboard</h1>
        <div class="hdr-sub">__GLOBAL_DATE__ &nbsp;·&nbsp; __N_COUNTRIES__ markets &nbsp;·&nbsp; Top __TOP_N__ colours per market</div>
    </div>
    <div class="graas-side">
      <div class="powered-lbl">Powered by</div>
      <img src="data:image/png;base64,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" alt="Graas" style="height:36px;width:auto">
  </div>
</div>
<div class="main">

  <div class="section-label">Select a market</div>
  <div class="grid">
    __CARDS__
  </div>

  <!-- Explainer — collapsible -->
  <div class="explainer-toggle" id="exp-toggle" onclick="toggleExp()">
    <h2>ℹ️ &nbsp;What is this? How does it work?</h2>
    <span class="arrow" id="exp-arrow">▼</span>
  </div>
  <div class="explainer-body" id="exp-body">
    <p>
      This tool measures <strong>price elasticity of demand</strong> — how much quantity sold changes
      when a product's price changes. It helps answer questions like:
      <em>If we discount this SKU by 10%, will the volume increase enough to offset the margin hit?
      Are customers more price-sensitive during a 11.11 sale vs. a regular day?
      Which categories respond most to price changes on TikTok vs. Shopee?</em>
    </p>
    <div class="steps">
      <div class="step">
        <div class="step-num">Step 1</div>
        <div class="step-title">Collect daily data</div>
        <div class="step-desc">Order-level data (SKU, channel, price, qty) from Jan 2024 across 6 SEA markets, enriched with product catalogue (Gender, Division, RBU, Colour).</div>
      </div>
      <div class="step">
        <div class="step-num">Step 2</div>
        <div class="step-title">Classify day types</div>
        <div class="step-desc">
          <strong>D-day</strong> — 10 mega sale campaigns:<br>
          <span style="font-family:monospace;font-size:10px;line-height:1.8">3.3 · 4.4 · 5.5 · 6.6 · 7.7 · 8.8 · 9.9 · 10.10 · 11.11 · 12.12</span><br><br>
          <strong>Special</strong> — mid-month &amp; payday:<br>
          <span style="font-family:monospace;font-size:10px;line-height:1.8">14th · 15th of every month (mid-month sale)<br>Last 4 days of every month (payday)</span><br><br>
          <strong>BAU</strong> — all remaining days
        </div>
      </div>
      <div class="step">
        <div class="step-num">Step 3</div>
        <div class="step-title">Log-log regression</div>
        <div class="step-desc">For each SKU we fit an OLS regression on log(price) vs. log(quantity). The slope is the elasticity coefficient. Requires ≥5 distinct price points.</div>
      </div>
      <div class="step">
        <div class="step-num">Step 4</div>
        <div class="step-title">Filter &amp; compare</div>
        <div class="step-desc">Slice by channel, gender, division, RBU, colour code, day type, and date range. All charts recalculate live — no server needed.</div>
    </div>
    <p style="margin-top:16px;font-weight:600;font-size:13px;color:var(--text)">How to read the elasticity number</p>
    <div class="interp">
      <div class="interp-card elastic">
        <div class="interp-title">Below −1 · Elastic</div>
        <div class="interp-desc">Demand drops more than proportionally when price rises. Customers are price-sensitive — discounting drives significant volume uplift.</div>
      </div>
      <div class="interp-card inelastic">
        <div class="interp-title">−1 to 0 · Inelastic</div>
        <div class="interp-desc">Demand holds up even as price changes. Brand loyalty or lack of alternatives. You have pricing power here.</div>
      </div>
      <div class="interp-card positive">
        <div class="interp-title">Positive · Anomalous</div>
        <div class="interp-desc">Higher price correlates with more demand — possible for luxury/status goods, or a data artefact (e.g. bundles). Interpret with caution.</div>
    </div>
    <div class="note">
      <strong>Data coverage:</strong> Top __TOP_N__ SKUs per market by volume · Free / zero-price units counted in volume stats but excluded from elasticity · ~94% of sales volume has full product metadata · ~6% (~10,600 SKUs) not matched to catalogue — see <em>unmapped_skus.csv</em>.
  </div>

</div>
<script>
function toggleExp(){
  const body=document.getElementById('exp-body');
  const toggle=document.getElementById('exp-toggle');
  const arrow=document.getElementById('exp-arrow');
  body.classList.toggle('open');
  toggle.classList.toggle('open');
}
</script>
</body>
</html>
"""

CARD_TEMPLATE = r"""
<a href="elasticity___CCY__.html" class="card">
  <div class="card-top">
    <div>
      <div class="card-ccy">__CCY__</div>
      <div class="card-country">__COUNTRY__</div>
    </div>
    <span class="tag">__N_CHANNELS__ ch</span>
  </div>
  <div class="card-stats">
    <div><div class="card-stat-lbl">Colours</div><div class="card-stat-val">__N_COLORS__</div></div>
    <div><div class="card-stat-lbl">Data Points</div><div class="card-stat-val">__N_RECORDS__</div></div>
  </div>
  <div class="card-date">__MIN_DATE__ → __MAX_DATE__</div>
  <div class="open-btn">Open Dashboard →</div>
</a>
"""


def get_plotly_js() -> str:
    import plotly as _plotly, os as _os
    js_path = _os.path.join(_os.path.dirname(_plotly.__file__),
                            "package_data", "plotly.min.js")
    with open(js_path, encoding="utf-8") as f:
        return f.read()


def generate_country_html(cdata: dict, plotly_js: str) -> str:
    data_json = json.dumps({k: v for k, v in cdata.items() if k != "ccy"
                            or True}, default=str, ensure_ascii=False)
    # strip keys not needed in JS
    js_data = {k: cdata[k] for k in
               ["channels","ch_labels","color_meta","def_colors",
                "genders","divisions","rbus","records",
                "min_date","max_date","n_colors","n_records"]}
    data_json = json.dumps(js_data, default=str, ensure_ascii=False)
    html = COUNTRY_HTML
    html = html.replace("__PLOTLY_JS__", plotly_js)
    html = html.replace("__DATA_JSON__",  data_json)
    html = html.replace("__CCY__",        cdata["ccy"])
    html = html.replace("__COUNTRY__",    cdata["country"])
    html = html.replace("__N_COLORS__",   str(cdata["n_colors"]))
    return html


def generate_index_html(all_data: dict, top_n: int) -> str:
    countries = all_data["countries"]
    total_recs = sum(d["n_records"] for d in countries.values())
    cards_html = ""
    for ccy, d in countries.items():
        c = CARD_TEMPLATE
        c = c.replace("__CCY__",        ccy)
        c = c.replace("__COUNTRY__",    d["country"])
        c = c.replace("__N_CHANNELS__", str(len(d["channels"])))
        c = c.replace("__N_COLORS__",   f'{d["n_colors"]:,}')
        c = c.replace("__N_RECORDS__",  f'{d["n_records"]:,}')
        c = c.replace("__MIN_DATE__",   d["min_date"])
        c = c.replace("__MAX_DATE__",   d["max_date"])
        cards_html += c

    html = INDEX_HTML
    html = html.replace("__GLOBAL_DATE__",   f'{all_data["min_date"]} → {all_data["max_date"]}')
    html = html.replace("__TOTAL_RECORDS__", f'{total_recs:,}')
    html = html.replace("__N_COUNTRIES__",   str(len(countries)))
    html = html.replace("__CARDS__",         cards_html)
    html = html.replace("__TOP_N__",         str(top_n))
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=None, help="Path to sales CSV")
    parser.add_argument("--meta",   default=None, help="Path to mapping XLSX")
    parser.add_argument("--outdir", default=None, help="Output directory (default: same as input CSV)")
    parser.add_argument("--top",    type=int, default=TOP_SKUS_PER_COUNTRY, help="Top N SKUs per country")
    args = parser.parse_args()

    # Auto-detect files
    if args.input:
        csv_path = Path(args.input)
    else:
        script_dir = Path(__file__).parent
        csvs = [f for f in script_dir.glob("*.csv") if "SKU data" in f.name or "sku" in f.name.lower()]
        if not csvs:
            csvs = list(script_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {script_dir}. Use --input.")
        csv_path = csvs[0]
        if len(csvs) > 1:
            print(f"Multiple CSVs found; using {csv_path.name}. Use --input to specify.")

    if args.meta:
        meta_path = args.meta
    else:
        # Look for xlsx alongside script
        script_dir = Path(__file__).parent
        xlsxs = list(script_dir.glob("*.xlsx")) + [
            Path("/Users/ajinkyapatil/Documents/mapping file for elasticity.xlsx")
        ]
        xlsxs = [x for x in xlsxs if x.exists()]
        if not xlsxs:
            raise FileNotFoundError("No XLSX mapping file found. Use --meta.")
        meta_path = str(xlsxs[0])

    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    all_data = process_all(str(csv_path), str(meta_path), args.top)

    print("\nGenerating HTML files …")
    plotly_js = get_plotly_js()

    for ccy, cdata in all_data["countries"].items():
        html = generate_country_html(cdata, plotly_js)
        out  = outdir / f"elasticity_{ccy}.html"
        out.write_text(html, encoding="utf-8")
        size_mb = len(html.encode()) / 1024 / 1024
        print(f"  {out.name}  ({size_mb:.1f} MB)")

    idx_html = generate_index_html(all_data, args.top)
    idx_out  = outdir / "elasticity_index.html"
    idx_out.write_text(idx_html, encoding="utf-8")
    print(f"  {idx_out.name}")

    print(f"\nDone. Open: {idx_out}")


if __name__ == "__main__":
    main()
