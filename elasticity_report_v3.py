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
    mf['ean'] = mf['ean'].astype(str).str.strip()
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
        <span style="font-weight:900;font-size:15px;letter-spacing:.1em;color:#1f2937">PUMA</span>
        <h1>Price Elasticity Dashboard</h1>
        <span class="ccy-badge">__CCY__</span>
        <span style="color:var(--muted);font-size:13px">__COUNTRY__</span>
      </div>
      <div class="hdr-sub" id="data-summary">Loading…</div>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;align-items:flex-end;gap:2px">
    <div style="font-size:9px;color:var(--muted);letter-spacing:.06em;text-transform:uppercase">Powered by</div>
    <div style="display:flex;align-items:center;gap:5px">
      <div style="width:22px;height:22px;background:linear-gradient(135deg,#22d3ee,#3b82f6);border-radius:5px;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:12px;color:#fff">G</div>
      <span style="font-size:15px;font-weight:700;color:#1f2937;letter-spacing:-.02em">raas</span>
    </div>
    <div style="font-size:9px;color:var(--muted)">AI-driven commerce analytics</div>
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
          <button onclick="resetDefault()">Top 1</button>
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
    <div class="card-title">Elasticity by SKU <span class="card-hint">click headers to sort</span>
      <span class="info" data-tip="Below −1 (Elastic): price-sensitive, demand drops sharply. −1 to 0 (Inelastic): demand holds. Positive: unusual — check data.">i</span>
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
  // overall OLS in log-log space, then draw as smooth power-law curve on linear axes
  const lps=pts.map(p=>Math.log(p.price)),lqs=pts.map(p=>Math.log(p.qty));
  const n=lps.length,slp=lps.reduce((a,b)=>a+b,0)/n,slq=lqs.reduce((a,b)=>a+b,0)/n;
  const num=lps.reduce((s,lp,i)=>s+(lp-slp)*(lqs[i]-slq),0);
  const den=lps.reduce((s,lp)=>s+(lp-slp)**2,0);
  if(den>0&&n>=5){
    const e=num/den,mn=Math.min(...pts.map(p=>p.price)),mx=Math.max(...pts.map(p=>p.price));
    // 100 evenly-spaced points → smooth curve in linear space
    const cx=Array.from({length:100},(_,i)=>mn+(mx-mn)*i/99);
    const cy=cx.map(p=>Math.exp(slq+e*(Math.log(p)-slp)));
    traces.push({type:'scatter',mode:'lines',name:`Overall e=${e.toFixed(2)}`,
      x:cx,y:cy,
      line:{color:'#dc2626',width:2.5,shape:'spline',smoothing:1.3},
      hovertemplate:`Price: %{x:,.0f}<br>Fitted Qty: %{y:,.0f}<br>Elasticity: ${e.toFixed(2)}<extra></extra>`});
  }
  pReact('ch-scatter',traces,{height:360,margin:{t:10,r:10,b:50,l:60},
    xaxis:{title:'Price',...GRID},
    yaxis:{title:'Total Quantity Sold',...GRID},
    legend:{orientation:'h',y:-0.22},...PLTBG,font:PLTFONT});
}

// ── Elasticity table ──────────────────────────────────────────────────────────
function badgeCls(e){ return e<-1?'b-el':e<0?'b-in':e<0.1?'b-un':'b-pos'; }
function badgeTxt(e){ return e<-1?'Elastic':e<0?'Inelastic':e<0.1?'Unitary':'Positive?'; }
function updTable(el){
  const s=[...el].sort((a,b)=>{
    const av=a[sortCol]??-Infinity,bv=b[sortCol]??-Infinity;
    return sortAsc?(av>bv?1:-1):(av<bv?1:-1);
  });
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
.hdr{background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 60%,#1d4ed8 100%);padding:20px 32px;color:#fff}
.hdr-inner{display:flex;align-items:center;justify-content:space-between;max-width:1040px;margin:0 auto}
.hdr-brand{display:flex;align-items:center;gap:16px}
.puma-wordmark{font-size:26px;font-weight:900;letter-spacing:.12em;color:#fff;line-height:1}
.hdr-titles h1{font-size:18px;font-weight:700;letter-spacing:-.01em;margin:0}
.hdr-sub{font-size:12px;margin-top:3px;opacity:.75}
.graas-brand{display:flex;flex-direction:column;align-items:flex-end;gap:4px}
.powered-lbl{font-size:10px;opacity:.6;letter-spacing:.06em;text-transform:uppercase}
.graas-logo{display:flex;align-items:center;gap:6px}
.graas-g{width:28px;height:28px;background:linear-gradient(135deg,#22d3ee,#3b82f6);border-radius:7px;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:15px;color:#fff;letter-spacing:0}
.graas-name{font-size:18px;font-weight:700;letter-spacing:-.02em;color:#fff}
.graas-tagline{font-size:10px;opacity:.55;text-align:right}

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
      <div class="puma-wordmark">PUMA</div>
      <div class="hdr-titles">
        <h1>Price Elasticity Dashboard</h1>
        <div class="hdr-sub">__GLOBAL_DATE__ &nbsp;·&nbsp; __N_COUNTRIES__ markets &nbsp;·&nbsp; Top __TOP_N__ colours per market</div>
      </div>
    </div>
    <div class="graas-brand">
      <div class="powered-lbl">Powered by</div>
      <div class="graas-logo">
        <div class="graas-g">G</div>
        <div class="graas-name">raas</div>
      </div>
      <div class="graas-tagline">AI-driven commerce analytics</div>
    </div>
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
        <div class="step-desc"><strong>D-day</strong> = mega campaigns (3.3–12.12) · <strong>Special</strong> = mid-month (14–15) &amp; payday (last 4 days) · <strong>BAU</strong> = everything else.</div>
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
    </div>
    <div class="note">
      <strong>Data coverage:</strong> Top __TOP_N__ SKUs per market by volume · Free / zero-price units counted in volume stats but excluded from elasticity · ~94% of sales volume has full product metadata · ~6% (~10,600 SKUs) not matched to catalogue — see <em>unmapped_skus.csv</em>.
    </div>
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
