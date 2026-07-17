#!/usr/bin/env python3
"""
Price Elasticity Analysis — FOC Stores  v1
Generates one self-contained HTML per country from Puma FOC Excel billing data.

Usage:
    python elasticity_report_foc.py
    python elasticity_report_foc.py --datadir /path/to/foc/files/ --outdir /path/to/output/
    python elasticity_report_foc.py --top 500
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
MIN_POINTS_FOR_ELASTICITY = 5
TOP_ARTICLES_PER_COUNTRY  = 1000
DEFAULT_SELECTED_ARTICLES = 5
MIN_RRP_RATIO             = 0.05   # price_per_unit / Retail pri must be ≥ 5%

# Rounding granularity per currency for OLS price-bucketing.
# Billing rounding artifacts (fractional per-unit prices from multi-qty txns)
# collapse into genuine price levels when we round to this many units.
PRICE_ROUND = {
    "IDR": 1000,    # e.g. 1,197,297.33 → 1,197,000
    "MYR": 1,       # e.g. 127.33 → 127
    "PHP": 100,     # e.g. 1,429.5 → 1,400
    "SGD": 1,       # e.g. 87.40 → 87
    "THB": 100,     # e.g. 1,247.5 → 1,200
    "VND": 10000,   # e.g. 421,296 → 420,000
}

DIVISION_MAP = {1: "Footwear", 2: "Apparel", 3: "Accessories"}
RBU_MAP = {
    10: "Teamsport",        20: "Motorsport",        30: "Golf",
    85: "Running",          86: "Accessories",      170: "Core",
   175: "Sportstyle Prime", 176: "Basketball",      180: "Kids",
   185: "Sportstyle Select",190: "Training",
}

COUNTRY_FILES = {
    "IDR": "FOC Puma - Indonesia.xlsx",
    "MYR": "FOC Puma - Malaysia.xlsx",
    "PHP": "FOC Puma - Philippines.xlsx",
    "SGD": "FOC Puma - Singapore.xlsx",
    "THB": "FOC Puma - Thailand.xlsx",
    "VND": "FOC Puma - Vietnam.xlsx",
}

COUNTRY_NAMES = {
    "IDR": "Indonesia", "MYR": "Malaysia",   "PHP": "Philippines",
    "SGD": "Singapore", "THB": "Thailand",   "VND": "Vietnam",
}
# ──────────────────────────────────────────────────────────────────────────────


def load_country(data_dir: Path, ccy: str, top_n: int) -> dict:
    fname = COUNTRY_FILES[ccy]
    fpath = data_dir / fname
    print(f"\n[{ccy}] Reading {fname} …")

    sheets = pd.read_excel(fpath, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    print(f"  Raw rows: {len(df):,}  sheets: {list(sheets.keys())}")

    # Select and rename required columns
    df = df[["Name", "Billing Date", "Style No.", "Description",
             "Division", "Reporting Business U", "Retail pri",
             "Billed qty", "Net invoice"]].copy()
    df.columns = ["store", "date", "style_no", "desc",
                  "div_code", "rbu_code", "retail_pri",
                  "billed_qty", "net_invoice"]

    # Type coercion
    df["date"]        = pd.to_datetime(df["date"], errors="coerce")
    df["style_no"]    = df["style_no"].astype(str).str.strip()
    df["desc"]        = df["desc"].fillna("").astype(str).str.strip()
    df["store"]       = df["store"].astype(str).str.strip()
    df["retail_pri"]  = pd.to_numeric(df["retail_pri"],  errors="coerce")
    df["billed_qty"]  = pd.to_numeric(df["billed_qty"],  errors="coerce")
    df["net_invoice"] = pd.to_numeric(df["net_invoice"], errors="coerce")
    df["div_code"]    = pd.to_numeric(df["div_code"],    errors="coerce")
    df["rbu_code"]    = pd.to_numeric(df["rbu_code"],    errors="coerce")

    df = df.dropna(subset=["date", "style_no", "billed_qty",
                            "net_invoice", "retail_pri", "store"])

    # Per-unit price and RRP ratio filter
    df["ppu"]       = df["net_invoice"] / df["billed_qty"]
    df["rrp_ratio"] = df["ppu"] / df["retail_pri"]

    before = len(df)
    df = df[(df["billed_qty"] > 0) & (df["rrp_ratio"] >= MIN_RRP_RATIO)]
    print(f"  After filter: {len(df):,} rows  (removed {before - len(df):,})")

    # Map Division and RBU codes → labels
    df["division"] = df["div_code"].map(DIVISION_MAP).fillna("Other")
    df["rbu"]      = df["rbu_code"].map(RBU_MAP).fillna("Other")

    # Round ppu to currency precision so different transactions at the same
    # genuine price level collapse into one bucket, while genuinely different
    # price points (e.g. SGD 15 vs SGD 30) stay separate.
    price_round = PRICE_ROUND.get(ccy, 1)
    df["ppu_r"] = (df["ppu"] / price_round).round() * price_round

    # Aggregate to (style_no, store, date, price_band) — keeps different price
    # points within the same store-day as separate records.
    agg = df.groupby(["style_no", "store", "date", "ppu_r"], as_index=False).agg(
        total_qty = ("billed_qty",  "sum"),
        total_rev = ("net_invoice", "sum"),
    )
    agg["avg_ppu"]  = agg["ppu_r"]   # use the rounded price; no blending
    agg["date_str"] = agg["date"].dt.strftime("%Y-%m-%d")

    # Top N articles by total volume
    art_vol  = agg.groupby("style_no")["total_qty"].sum().sort_values(ascending=False)
    eligible = art_vol[art_vol > 10]
    print(f"  [{ccy}] total distinct articles: {len(art_vol):,}  "
          f"|  qty>10: {len(eligible):,}  |  capping at top {top_n}")
    top_arts = art_vol.head(top_n).index.tolist()

    # Article metadata — pick description from highest-qty row per style
    art_desc = (df.sort_values("billed_qty", ascending=False)
                  .drop_duplicates(subset="style_no")
                  .set_index("style_no")["desc"])
    art_div  = df.groupby("style_no")["division"].first()
    art_rbu  = df.groupby("style_no")["rbu"].first()

    meta = []
    for cn in top_arts:
        meta.append({
            "cn":       cn,
            "name":     art_desc.get(cn, cn),
            "division": art_div.get(cn, "Other"),
            "rbu":      art_rbu.get(cn, "Other"),
        })

    stores    = sorted(df["store"].unique().tolist())
    divisions = sorted(set(m["division"] for m in meta if m["division"] != "Other"))
    rbus      = sorted(set(m["rbu"]      for m in meta if m["rbu"]      != "Other"))
    if any(m["division"] == "Other" for m in meta): divisions.append("Other")
    if any(m["rbu"]      == "Other" for m in meta): rbus.append("Other")

    agg_top = agg[agg["style_no"].isin(top_arts)]
    records = [
        {"cn": r.style_no, "st": r.store, "d": r.date_str,
         "p": r.avg_ppu,  "tq": int(r.total_qty)}
        for r in agg_top.itertuples(index=False)
    ]

    min_date = agg["date"].min().strftime("%Y-%m-%d")
    max_date = agg["date"].max().strftime("%Y-%m-%d")

    print(f"  {ccy}: {len(top_arts)} articles, {len(records):,} records, "
          f"{len(stores)} stores,  {min_date} → {max_date}")

    return {
        "ccy":         ccy,
        "country":     COUNTRY_NAMES.get(ccy, ccy),
        "price_round": PRICE_ROUND.get(ccy, 1),
        "stores":      stores,
        "color_meta":  meta,
        "def_colors":  top_arts[:DEFAULT_SELECTED_ARTICLES],
        "divisions":   divisions,
        "rbus":        rbus,
        "records":     records,
        "min_date":    min_date,
        "max_date":    max_date,
        "n_colors":    len(top_arts),
        "n_records":   len(records),
    }


# ── Country HTML template ──────────────────────────────────────────────────────
COUNTRY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Price Elasticity (FOC) — __CCY__</title>
<script>__PLOTLY_JS__</script>
<style>
:root{--bg:#f8f9fa;--card:#fff;--border:#dee2e6;--primary:#2563eb;
  --text:#1f2937;--muted:#6b7280}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text)}

.hdr{background:var(--card);border-bottom:1px solid var(--border);padding:12px 24px;display:flex;align-items:center;justify-content:space-between}
.hdr-left{display:flex;align-items:center;gap:14px}
.back-btn{padding:5px 12px;border:1px solid var(--border);border-radius:6px;background:#fff;font-size:12px;cursor:pointer;color:var(--muted);text-decoration:none;display:inline-flex;align-items:center;gap:4px}
.back-btn:hover{border-color:var(--primary);color:var(--primary)}
.hdr h1{font-size:17px;font-weight:600}
.hdr-sub{color:var(--muted);font-size:12px;margin-top:2px}
.ccy-badge{padding:3px 10px;background:#eff6ff;border:1px solid #bfdbfe;border-radius:20px;font-size:13px;font-weight:700;color:var(--primary)}
.foc-badge{padding:3px 10px;background:#fef3c7;border:1px solid #fde68a;border-radius:20px;font-size:11px;font-weight:700;color:#92400e;margin-left:6px}

.filters{background:var(--card);border-bottom:1px solid var(--border);padding:10px 24px;display:flex;gap:18px;align-items:flex-end;flex-wrap:wrap}
.fg{display:flex;flex-direction:column;gap:4px}
.fl{font-size:10px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.info{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;border-radius:50%;background:#e5e7eb;font-size:9px;cursor:help;color:var(--muted);font-style:normal;position:relative}
.info:hover::after{content:attr(data-tip);position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);background:#1f2937;color:#fff;font-size:11px;padding:6px 10px;border-radius:6px;white-space:nowrap;z-index:99;pointer-events:none;max-width:280px;white-space:normal;line-height:1.4;font-weight:400}

.tgl{padding:4px 10px;border:1px solid var(--border);border-radius:5px;font-size:11px;font-weight:500;cursor:pointer;background:#fff;color:var(--muted);transition:all .1s}
.tgl.on{background:#eff6ff;border-color:#bfdbfe;color:var(--primary)}
.tgl:hover{border-color:var(--primary)}

/* Store filter — scrollable chip row */
.store-row{display:flex;flex-wrap:wrap;gap:5px;max-width:520px}

/* RBU dropdown */
.rbu-wrap{position:relative}
.rbu-btn{display:flex;align-items:center;gap:6px;padding:5px 10px;border:1px solid var(--border);border-radius:6px;background:#fff;font-size:12px;cursor:pointer;min-width:130px;justify-content:space-between}
.rbu-btn:hover{border-color:var(--primary)}
.rbu-drop{display:none;position:absolute;top:calc(100% + 4px);left:0;background:#fff;border:1px solid var(--border);border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.1);z-index:50;min-width:200px;max-height:260px;overflow-y:auto;padding:6px}
.rbu-drop.open{display:block}
.rbu-item{display:flex;align-items:center;gap:8px;padding:5px 8px;border-radius:5px;font-size:12px;cursor:pointer}
.rbu-item:hover{background:#f3f4f6}
.rbu-actions{display:flex;gap:6px;padding:6px 8px 2px;border-bottom:1px solid var(--border);margin-bottom:4px}
.rbu-actions button{font-size:11px;padding:2px 8px;border:1px solid var(--border);border-radius:4px;cursor:pointer;background:#fff}
.rbu-actions button:hover{background:#f3f4f6}

/* Article dropdown */
.sku-wrap{position:relative}
.sku-btn{display:flex;align-items:center;gap:6px;padding:5px 10px;border:1px solid var(--border);border-radius:6px;background:#fff;font-size:12px;cursor:pointer;min-width:180px;justify-content:space-between}
.sku-btn:hover{border-color:var(--primary)}
.sku-drop{display:none;position:absolute;top:calc(100% + 4px);left:0;background:#fff;border:1px solid var(--border);border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.1);z-index:50;width:320px}
.sku-drop.open{display:block}
.sku-search{width:100%;padding:8px 10px;border:none;border-bottom:1px solid var(--border);font-size:12px;outline:none}
.sku-list{max-height:260px;overflow-y:auto;padding:4px}
.sku-item{display:flex;align-items:flex-start;gap:8px;padding:7px 8px;border-radius:5px;font-size:12px;cursor:pointer;width:100%}
.sku-item:hover{background:#f3f4f6}
.sku-item input{margin-top:2px;flex-shrink:0}
.sku-name{font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px}
.sku-sub{font-size:10px;color:var(--muted);font-family:monospace}
.sku-tags{display:flex;flex-wrap:wrap;gap:3px;margin-top:3px}
.sku-tag{font-size:10px;padding:1px 5px;background:#f3f4f6;border-radius:3px;color:var(--muted)}
.sku-actions{display:flex;gap:6px;padding:6px 8px;border-top:1px solid var(--border)}
.sku-actions button{font-size:11px;padding:2px 8px;border:1px solid var(--border);border-radius:4px;cursor:pointer;background:#fff}
.sku-actions button:hover{background:#f3f4f6}

.dr{display:flex;align-items:center;gap:6px}
.dr input{padding:4px 8px;border:1px solid var(--border);border-radius:6px;font-size:12px;color:var(--text)}
.dr span{color:var(--muted);font-size:12px}

.section-desc{padding:8px 24px;background:#f9fafb;border-bottom:1px solid var(--border);font-size:11px;color:var(--muted)}

.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border-bottom:1px solid var(--border)}
.stat{padding:14px 20px;border-right:1px solid var(--border)}
.stat:last-child{border-right:none}
.stat-lbl{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.stat-val{font-size:24px;font-weight:700;color:var(--text);margin:4px 0}
.stat-sub{font-size:11px;color:var(--muted);line-height:1.4}

.grid{display:grid;grid-template-columns:1fr 1fr;gap:0;border-bottom:1px solid var(--border)}
.card{border-right:1px solid var(--border);border-bottom:1px solid var(--border);padding:14px}
.card.full{grid-column:1/-1;border-right:none}
.card-title{font-size:13px;font-weight:600;color:var(--text);margin-bottom:8px;display:flex;align-items:center;gap:6px}
.card-hint{font-size:10px;font-weight:400;color:var(--muted)}
.card-body{min-height:200px}

.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:600}
.b-el{background:#fee2e2;color:#dc2626}
.b-in{background:#dcfce7;color:#16a34a}
.b-un{background:#fef9c3;color:#ca8a04}
.b-pos{background:#fef3c7;color:#d97706}
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #f3f4f6}
th{font-weight:600;color:var(--muted);font-size:11px;white-space:nowrap;cursor:pointer;user-select:none;background:#fafafa;position:sticky;top:0}
th:hover{color:var(--primary)}
tr:hover td{background:#f9fafb}
.no-data{padding:40px;text-align:center;color:var(--muted);font-size:13px}
.err{padding:20px;text-align:center;color:#9ca3af;font-size:13px}

#data-summary{font-size:11px;color:var(--muted)}
</style>
</head>
<body>
<div class="hdr">
  <div class="hdr-left">
    <a href="elasticity_foc_index.html" class="back-btn">← All Markets</a>
    <div>
      <div style="display:flex;align-items:center;gap:6px">
        <h1>Price Elasticity — __COUNTRY__</h1>
        <span class="ccy-badge">__CCY__</span>
        <span class="foc-badge">FOC</span>
      </div>
      <div class="hdr-sub" id="data-summary">Loading…</div>
    </div>
  </div>
  <img src="data:image/webp;base64,__PUMA_LOGO__" alt="PUMA" style="height:44px;width:auto">
</div>

<div class="filters">
  <div class="fg">
    <div class="fl">Store <span class="info" data-tip="Toggle individual FOC stores on/off">i</span></div>
    <div class="store-row" id="store-filter"></div>
  </div>

  <div class="fg">
    <div class="fl">Division</div>
    <div id="division-filter" style="display:flex;flex-wrap:wrap;gap:5px"></div>
  </div>

  <div class="fg">
    <div class="fl">RBU</div>
    <div class="rbu-wrap">
      <button class="rbu-btn" onclick="toggleRbuDrop()">
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

  <div class="fg">
    <div class="fl">Articles (Styles) <span class="info" data-tip="Search by style number or product name">i</span></div>
    <div class="sku-wrap">
      <button class="sku-btn" onclick="toggleSkuDrop()">
        <span id="sku-lbl">5 selected</span><span>▾</span>
      </button>
      <div class="sku-drop" id="sku-drop">
        <input class="sku-search" id="sku-q" placeholder="Search style no. or name…" oninput="renderSkuList(this.value)">
        <div class="sku-list" id="sku-list"></div>
        <div class="sku-actions">
          <button onclick="selectAll()">Select all</button>
          <button onclick="clearAll()">Clear</button>
          <button onclick="resetDefault()">Top 5</button>
        </div>
      </div>
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
  Top __N_COLORS__ articles by volume · FOC physical retail stores · Each article aggregated to daily (store × date) price points ·
  Division / RBU filters narrow the article list above.
</div>

<div class="stats">
  <div class="stat"><div class="stat-lbl">Articles w/ Elasticity</div><div class="stat-val" id="s-skus">—</div><div class="stat-sub">Articles (styles) with ≥5 unique price observations in current filter view</div></div>
  <div class="stat"><div class="stat-lbl">Avg Elasticity</div><div class="stat-val" id="s-el">—</div><div class="stat-sub"><strong>Below −1</strong> = elastic · <strong>−1 to 0</strong> = inelastic · <strong>Positive</strong> = anomalous</div></div>
  <div class="stat"><div class="stat-lbl">Total Units Sold</div><div class="stat-val" id="s-qty">—</div><div class="stat-sub">Sum across selected articles (styles), stores &amp; date range</div></div>
  <div class="stat"><div class="stat-lbl">Avg Selling Price</div><div class="stat-val" id="s-price">—</div><div class="stat-sub">Weighted average net selling price (weighted by qty sold)</div></div>
</div>

<div class="grid">
  <div class="card">
    <div class="card-title">Price vs. Quantity
      <span class="info" data-tip="Each dot = one unique price point across all stores for an article. The dotted line is the estimated demand curve (log-log OLS). Downward slope = higher price → lower demand.">i</span>
    </div>
    <div class="card-body"><div id="ch-scatter"></div></div>
  </div>
  <div class="card">
    <div class="card-title">Elasticity by Article <span class="card-hint">click headers to sort</span>
      <span class="info" data-tip="Below −1 (Elastic): price-sensitive, demand drops sharply. −1 to 0 (Inelastic): demand holds. Positive: unusual — check data.">i</span>
      <button onclick="exportTableCSV()" style="margin-left:auto;padding:3px 10px;font-size:11px;font-weight:500;border:1px solid var(--border);border-radius:5px;background:#f9fafb;cursor:pointer;color:var(--text)" title="Download table as CSV">⬇ Export CSV</button>
    </div>
    <div class="card-body" id="tbl-wrap"></div>
  </div>
  <div class="card full">
    <div class="card-title">Price &amp; Quantity Over Time
      <span class="info" data-tip="Blue line = avg selling price (left axis). Bars = total units sold (right axis).">i</span>
    </div>
    <div class="card-body"><div id="ch-ts"></div></div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;
const DATA_CCY = '__CCY__';

// ── State ─────────────────────────────────────────────────────────────────────
let selStores   = new Set(DATA.stores);
let selDivision = new Set(DATA.divisions);
let selRBU      = new Set(DATA.rbus);
let sel         = new Set(DATA.def_colors.map(String));
let d0          = DATA.min_date;
let d1          = DATA.max_date;
let sortCol     = 'total_qty';
let sortAsc     = false;

// ── Plotly constants ──────────────────────────────────────────────────────────
const PLTCFG  = {responsive:true,displayModeBar:false};
const PLTFONT = {family:"-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif",size:11};
const PLTBG   = {paper_bgcolor:'white',plot_bgcolor:'white'};
const GRID    = {showgrid:true,gridcolor:'#f3f4f6'};

// ── Article lookup helpers ────────────────────────────────────────────────────
const COLOR_MAP = {};
DATA.color_meta.forEach(m=>{ COLOR_MAP[m.cn]=m; });
function colorInfo(cn){ return COLOR_MAP[cn]||{cn,name:cn,division:'',rbu:''}; }

// ── Visible articles (intersection of metadata filters) ────────────────────
function visibleColors(){
  return DATA.color_meta
    .filter(m=>
      selDivision.has(m.division) &&
      selRBU.has(m.rbu)
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
    `${DATA.n_records.toLocaleString()} daily aggregates · top ${DATA.n_colors} articles · ${DATA.min_date} → ${DATA.max_date}`;

  renderStoreFilter();
  renderMetaFilter('division-filter', DATA.divisions, selDivision, toggleDivision);
  renderRbuList();
  updRbuBtn();
  renderSkuList('');

  document.addEventListener('click', e=>{
    if(!document.querySelector('.sku-wrap').contains(e.target))
      document.getElementById('sku-drop').classList.remove('open');
    if(!document.querySelector('.rbu-wrap').contains(e.target))
      document.getElementById('rbu-drop').classList.remove('open');
  });

  updateAll();
}

// ── Store filter ──────────────────────────────────────────────────────────────
function renderStoreFilter(){
  const wrap=document.getElementById('store-filter');
  wrap.innerHTML=DATA.stores.map(st=>`
    <button class="tgl ${selStores.has(st)?'on':''}"
            data-st="${st.replace(/"/g,'&quot;')}" onclick="toggleStore('${st.replace(/'/g,"\\'")}')">
      ${st.length>22?st.slice(0,20)+'…':st}
    </button>
  `).join('');
}
function toggleStore(st){
  if(selStores.has(st)){ if(selStores.size===1)return; selStores.delete(st); }
  else selStores.add(st);
  document.querySelector(`.tgl[data-st="${st.replace(/"/g,'&quot;')}"]`).classList.toggle('on',selStores.has(st));
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
function toggleDivision(v){ toggleMeta(v,selDivision,'division-filter',()=>renderMetaFilter('division-filter',DATA.divisions,selDivision,toggleDivision)); }

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

// ── Article multi-select ──────────────────────────────────────────────────────
function renderSkuList(q){
  const lq=q.toLowerCase();
  const vis=visibleColorSet();
  const items=DATA.color_meta.filter(m=>
    vis.has(m.cn) &&
    (!q || m.cn.toLowerCase().includes(lq) || m.name.toLowerCase().includes(lq))
  );
  document.getElementById('sku-list').innerHTML=items.length===0
    ?'<div style="padding:20px;text-align:center;color:var(--muted);font-size:12px">No articles match current filters</div>'
    :items.map(m=>`
      <label class="sku-item">
        <input type="checkbox" value="${m.cn}" ${sel.has(m.cn)?'checked':''} onchange="onSku('${m.cn}',this.checked)">
        <div>
          <div class="sku-name">${m.name||m.cn}</div>
          <div class="sku-sub">${m.cn}</div>
          <div class="sku-tags">
            ${m.division&&m.division!=='Other'?`<span class="sku-tag">${m.division}</span>`:''}
            ${m.rbu&&m.rbu!=='Other'?`<span class="sku-tag">${m.rbu}</span>`:''}
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
    :`${active.length} articles selected`;
  const el=document.getElementById('sku-lbl');
  el.textContent=lbl.length>35?lbl.slice(0,33)+'…':lbl;
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
    selStores.has(r.st) &&
    sel.has(r.cn) &&
    vis.has(r.cn) &&
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
  // Round price to nearest PRICE_ROUND units before OLS to collapse
  // billing rounding artifacts (multi-qty fractional per-unit prices)
  // into genuine markdown price levels.
  const pr=DATA.price_round||1;
  const m={};
  recs.forEach(r=>{
    if(!m[r.cn])m[r.cn]={buckets:{},totalQty:0,rev:0};
    m[r.cn].totalQty+=r.tq;
    if(r.tq>0){
      const pb=Math.round(r.p/pr)*pr;
      m[r.cn].buckets[pb]=(m[r.cn].buckets[pb]||0)+r.tq;
      m[r.cn].rev+=r.p*r.tq;
    }
  });
  return Object.entries(m).map(([cn,d])=>{
    const prices=Object.keys(d.buckets).map(Number);
    const qtys=prices.map(p=>d.buckets[p]);
    const info=colorInfo(cn);
    return {
      cn, name:info.name||cn,
      division:info.division||'', rbu:info.rbu||'',
      elasticity:ols(prices,qtys),
      avg_price:d.totalQty>0?d.rev/d.totalQty:0,
      total_qty:d.totalQty, data_points:prices.length
    };
  }).filter(r=>r.elasticity!==null);
}

// ── Plotly wrapper ────────────────────────────────────────────────────────────
function pReact(id,traces,layout,cfg){
  try{ Plotly.react(id,traces,layout,cfg||PLTCFG); }
  catch(e){
    const el=document.getElementById(id);
    if(el&&!el.querySelector('.err'))
      el.innerHTML=`<div class="err">⚠️ ${e.message}</div>`;
  }
}

// ── Update all ────────────────────────────────────────────────────────────────
function updateAll(){
  const f=filtered(),el=elByColor(f);
  updStats(f,el); updTable(el); updScatter(f,el); updTS(f);
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
    if(r.tq===0)return;
    const p=Math.round(r.p),key=`${r.cn}||${p}`;
    if(!byPriceKey[key])byPriceKey[key]={cn:r.cn,price:p,qty:0,days:0};
    byPriceKey[key].qty+=r.tq; byPriceKey[key].days+=1;
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
    x:xs,y:ys,text:ts,hoverinfo:'text',marker:{color:'#f59e0b',opacity:.5,size:6}}];
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
  const cols=['Style No','Product','Division','RBU','Elasticity','Avg Price','Total Qty','Data Points'];
  const rows=lastTableData.map(r=>[
    r.cn, `"${(r.name||'').replace(/"/g,'""')}"`,
    r.division||'', r.rbu||'',
    r.elasticity.toFixed(2), Math.round(r.avg_price), r.total_qty, r.data_points
  ].join(','));
  const csv=[cols.join(','),...rows].join('\n');
  const a=document.createElement('a');
  a.href='data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
  a.download=`elasticity_foc_${DATA_CCY||'export'}.csv`;
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
    ?'<div class="no-data">No articles with ≥5 data points in current selection</div>'
    :`<div class="tbl-wrap"><table>
      <thead><tr>
        <th onclick="sort('name')">Product${si('name')}</th>
        <th onclick="sort('cn')">Style No${si('cn')}</th>
        <th onclick="sort('division')">Division${si('division')}</th>
        <th onclick="sort('rbu')">RBU${si('rbu')}</th>
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
     marker:{color:'#fde68a'},yaxis:'y2',hovertemplate:'%{x}<br>Qty: %{y:,}<extra></extra>'},
    {type:'scatter',mode:'lines+markers',name:'Avg Price',x:dates,
     y:dates.map(d=>byDate[d].qty>0?byDate[d].rev/byDate[d].qty:null),
     line:{color:'#f59e0b',width:2},marker:{size:3},hovertemplate:'%{x}<br>Price: %{y:.2f}<extra></extra>'},
  ],{height:280,margin:{t:10,r:64,b:50,l:60},xaxis:{showgrid:false},
    yaxis:{title:'Avg Price',...GRID},
    yaxis2:{title:'Total Qty',overlaying:'y',side:'right',showgrid:false},
    legend:{orientation:'h',y:-0.28},...PLTBG,font:PLTFONT},PLTCFG);
}

// ── Start ─────────────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>
"""

# ── FOC Index page ─────────────────────────────────────────────────────────────
FOC_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Price Elasticity — FOC Stores</title>
<style>
:root{--bg:#f0f4f8;--card:#fff;--border:#dee2e6;--primary:#d97706;--text:#1f2937;--muted:#6b7280}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}

.hdr{background:#fff;border-bottom:2px solid #e5e7eb;padding:14px 32px}
.hdr-inner{display:flex;align-items:center;justify-content:space-between;max-width:1040px;margin:0 auto}
.hdr-brand{display:flex;align-items:center;gap:20px}
.hdr-divider{width:1px;height:36px;background:#e5e7eb}
.hdr-titles h1{font-size:17px;font-weight:700;color:#1f2937;letter-spacing:-.01em;margin:0}
.hdr-sub{font-size:11px;margin-top:3px;color:#6b7280}
.foc-pill{display:inline-block;padding:3px 10px;background:#fef3c7;border:1px solid #fde68a;border-radius:20px;font-size:11px;font-weight:700;color:#92400e;margin-left:8px;vertical-align:middle}
.graas-side{display:flex;flex-direction:column;align-items:flex-end;gap:4px}
.graas-pill{background:#0f172a;border-radius:8px;padding:6px 12px;display:flex;flex-direction:column;align-items:center;gap:3px}
.powered-lbl{font-size:9px;color:rgba(255,255,255,.55);letter-spacing:.08em;text-transform:uppercase}
.graas-tagline{font-size:9px;color:#9ca3af;text-align:right}

.main{max-width:1040px;margin:0 auto;padding:24px 24px 40px}
.section-label{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:28px}
@media(max-width:680px){.grid{grid-template-columns:1fr 1fr}}
@media(max-width:420px){.grid{grid-template-columns:1fr}}

.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px 20px;text-decoration:none;color:inherit;display:block;transition:all .15s;position:relative;overflow:hidden}
.card:hover{border-color:var(--primary);box-shadow:0 4px 16px rgba(217,119,6,.15);transform:translateY(-2px)}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--primary);opacity:0;transition:opacity .15s}
.card:hover::before{opacity:1}
.card-top{display:flex;align-items:flex-start;justify-content:space-between}
.card-ccy{font-size:28px;font-weight:800;color:var(--primary);line-height:1}
.card-country{font-size:13px;font-weight:600;color:var(--text);margin-top:3px}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;background:#fef3c7;color:var(--primary)}
.card-stats{margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:6px 12px}
.card-stat-lbl{font-size:10px;color:var(--muted)}
.card-stat-val{font-size:13px;font-weight:600;color:var(--text)}
.card-date{font-size:10px;color:var(--muted);margin-top:10px;padding-top:10px;border-top:1px solid var(--border)}
.open-btn{margin-top:12px;display:flex;align-items:center;justify-content:center;gap:6px;padding:6px;border:1px solid var(--border);border-radius:6px;font-size:12px;font-weight:500;color:var(--primary);background:#fffbeb;transition:all .12s}
.card:hover .open-btn{background:var(--primary);color:#fff;border-color:var(--primary)}

.back-row{margin-bottom:20px}
.back-btn{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border:1px solid var(--border);border-radius:6px;background:#fff;font-size:12px;cursor:pointer;color:var(--muted);text-decoration:none}
.back-btn:hover{border-color:var(--primary);color:var(--primary)}

.explainer-toggle{display:flex;align-items:center;justify-content:space-between;cursor:pointer;padding:14px 18px;background:var(--card);border:1px solid var(--border);border-radius:10px;margin-bottom:0;user-select:none}
.explainer-toggle:hover{border-color:#94a3b8}
.explainer-toggle h2{font-size:14px;font-weight:600;color:var(--text)}
.explainer-toggle .arrow{font-size:13px;color:var(--muted);transition:transform .2s}
.explainer-toggle.open .arrow{transform:rotate(180deg)}
.explainer-body{background:var(--card);border:1px solid var(--border);border-top:none;border-radius:0 0 10px 10px;padding:20px 22px;display:none}
.explainer-body.open{display:block}
.explainer-body p{font-size:13px;color:#374151;line-height:1.7;margin-bottom:10px}
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}
@media(max-width:700px){.steps{grid-template-columns:1fr 1fr}}
.step{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:12px}
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
      <img src="data:image/webp;base64,__PUMA_LOGO__" alt="PUMA" style="height:52px;width:auto">
      <div class="hdr-divider"></div>
      <div class="hdr-titles">
        <h1>Price Elasticity Dashboard <span class="foc-pill">FOC Stores</span></h1>
        <div class="hdr-sub">__GLOBAL_DATE__ &nbsp;·&nbsp; __N_COUNTRIES__ markets &nbsp;·&nbsp; Top __TOP_N__ articles per market</div>
      </div>
    </div>
    <div class="graas-side">
      <div class="powered-lbl">Powered by</div>
      <img src="data:image/png;base64,__GRAAS_LOGO__" alt="Graas" style="height:36px;width:auto">
    </div>
  </div>
</div>
<div class="main">
  <div class="back-row">
    <a href="elasticity_index.html" class="back-btn">← Marketplace Dashboard</a>
  </div>

  <div class="section-label">Select a market</div>
  <div class="grid">
    __CARDS__
  </div>

  <div class="explainer-toggle" id="exp-toggle" onclick="toggleExp()">
    <h2>ℹ️ &nbsp;What is this? How does it work?</h2>
    <span class="arrow" id="exp-arrow">▼</span>
  </div>
  <div class="explainer-body" id="exp-body">
    <p>
      This tool measures <strong>price elasticity of demand</strong> across Puma
      <strong>Factory Outlet Centre (FOC)</strong> physical retail stores in SEA —
      how much quantity sold changes when a product's price changes.
      Unlike the marketplace version, FOC data comes from billing records rather than
      online orders, and does not have campaign sale days (D-days).
    </p>
    <div class="steps">
      <div class="step">
        <div class="step-num">Step 1</div>
        <div class="step-title">Collect billing data</div>
        <div class="step-desc">Billing-line data (Style No., store, net price, qty) from Puma FOC stores across 6 SEA markets. Division and RBU come directly from the billing system.</div>
      </div>
      <div class="step">
        <div class="step-num">Step 2</div>
        <div class="step-title">Clean &amp; filter</div>
        <div class="step-desc">Returns (negative qty) and near-free promotional sales (price &lt; 5% of RRP) are excluded. Per-unit price = Net invoice ÷ Billed qty.</div>
      </div>
      <div class="step">
        <div class="step-num">Step 3</div>
        <div class="step-title">Log-log regression</div>
        <div class="step-desc">For each article (style) we fit an OLS regression on log(price) vs. log(quantity), aggregated to daily (store × date) level. The slope is the elasticity coefficient. Requires ≥5 distinct price points.</div>
      </div>
      <div class="step">
        <div class="step-num">Step 4</div>
        <div class="step-title">Filter &amp; compare</div>
        <div class="step-desc">Slice by store, division, RBU, article, and date range. All charts recalculate live — no server needed.</div>
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
        <div class="interp-desc">Higher price correlates with more demand — possible for luxury/status goods, or a data artefact. Interpret with caution.</div>
      </div>
    </div>
    <div class="note">
      <strong>Data coverage:</strong> Top __TOP_N__ articles per market by volume · Sales at price &lt; 5% of RRP and returns excluded · Price elasticity in FOC context reflects markdown behaviour across stores and dates.
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

FOC_CARD_TEMPLATE = r"""
<a href="elasticity_foc___CCY__.html" class="card">
  <div class="card-top">
    <div>
      <div class="card-ccy">__CCY__</div>
      <div class="card-country">__COUNTRY__</div>
    </div>
    <span class="tag">__N_STORES__ stores</span>
  </div>
  <div class="card-stats">
    <div><div class="card-stat-lbl">Articles</div><div class="card-stat-val">__N_COLORS__</div></div>
    <div><div class="card-stat-lbl">Data Points</div><div class="card-stat-val">__N_RECORDS__</div></div>
  </div>
  <div class="card-date">__MIN_DATE__ → __MAX_DATE__</div>
  <div class="open-btn">Open Dashboard →</div>
</a>
"""


def get_plotly_js() -> str:
    import plotly as _plotly
    import os as _os
    js_path = _os.path.join(_os.path.dirname(_plotly.__file__),
                            "package_data", "plotly.min.js")
    with open(js_path, encoding="utf-8") as f:
        return f.read()


def get_logo_b64(path: str) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_country_html(cdata: dict, plotly_js: str, puma_logo: str) -> str:
    js_data = {k: cdata[k] for k in
               ["stores", "color_meta", "def_colors",
                "divisions", "rbus", "records",
                "min_date", "max_date", "n_colors", "n_records"]}
    data_json = json.dumps(js_data, default=str, ensure_ascii=False)
    html = COUNTRY_HTML
    html = html.replace("__PLOTLY_JS__", plotly_js)
    html = html.replace("__DATA_JSON__",  data_json)
    html = html.replace("__CCY__",        cdata["ccy"])
    html = html.replace("__COUNTRY__",    cdata["country"])
    html = html.replace("__N_COLORS__",   str(cdata["n_colors"]))
    html = html.replace("__PUMA_LOGO__",  puma_logo)
    return html


def generate_index_html(all_data: dict, top_n: int,
                        puma_logo: str, graas_logo: str) -> str:
    countries = all_data["countries"]
    cards_html = ""
    for ccy, d in countries.items():
        c = FOC_CARD_TEMPLATE
        c = c.replace("__CCY__",        ccy)
        c = c.replace("__COUNTRY__",    d["country"])
        c = c.replace("__N_STORES__",   str(len(d["stores"])))
        c = c.replace("__N_COLORS__",   f'{d["n_colors"]:,}')
        c = c.replace("__N_RECORDS__",  f'{d["n_records"]:,}')
        c = c.replace("__MIN_DATE__",   d["min_date"])
        c = c.replace("__MAX_DATE__",   d["max_date"])
        cards_html += c

    html = FOC_INDEX_HTML
    html = html.replace("__GLOBAL_DATE__", f'{all_data["min_date"]} → {all_data["max_date"]}')
    html = html.replace("__N_COUNTRIES__", str(len(countries)))
    html = html.replace("__CARDS__",       cards_html)
    html = html.replace("__TOP_N__",       str(top_n))
    html = html.replace("__PUMA_LOGO__",   puma_logo)
    html = html.replace("__GRAAS_LOGO__",  graas_logo)
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default=None, help="Directory containing FOC Excel files")
    parser.add_argument("--outdir",  default=None, help="Output directory for HTML files")
    parser.add_argument("--top",     type=int, default=TOP_ARTICLES_PER_COUNTRY)
    args = parser.parse_args()

    # Default data dir: look for FOC files near the script
    script_dir = Path(__file__).parent
    if args.datadir:
        data_dir = Path(args.datadir)
    else:
        # Try common locations
        candidates = [
            script_dir / "FOC Puma files" / "Final files",
            script_dir,
        ]
        data_dir = next((p for p in candidates if (p / "FOC Puma - Indonesia.xlsx").exists()), script_dir)

    outdir = Path(args.outdir) if args.outdir else script_dir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Data dir : {data_dir}")
    print(f"Output   : {outdir}")
    print(f"Top N    : {args.top}")

    # Load logos from the existing marketplace HTML if available
    # (reuse same base64 blobs to keep files consistent)
    puma_logo  = ""
    graas_logo = ""
    existing_html = outdir / "elasticity_IDR.html"
    if existing_html.exists():
        txt = existing_html.read_text(encoding="utf-8")
        import re
        m = re.search(r'src="data:image/webp;base64,([^"]+)"', txt)
        if m: puma_logo = m.group(1)
        m = re.search(r'src="data:image/png;base64,([^"]+)"', txt)
        if m: graas_logo = m.group(1)
        print("Logos extracted from existing elasticity_IDR.html")
    else:
        print("WARNING: elasticity_IDR.html not found — logos will be blank. "
              "Run elasticity_report_v3.py first or place it in the same output dir.")

    all_countries = {}
    global_min, global_max = "9999-12-31", "0000-01-01"

    for ccy in COUNTRY_FILES:
        fpath = data_dir / COUNTRY_FILES[ccy]
        if not fpath.exists():
            print(f"  SKIP {ccy} — file not found: {fpath}")
            continue
        try:
            cdata = load_country(data_dir, ccy, args.top)
            all_countries[ccy] = cdata
            if cdata["min_date"] < global_min: global_min = cdata["min_date"]
            if cdata["max_date"] > global_max: global_max = cdata["max_date"]
        except Exception as e:
            print(f"  ERROR loading {ccy}: {e}")
            raise

    all_data = {"countries": all_countries, "min_date": global_min, "max_date": global_max}

    print("\nGenerating HTML files …")
    plotly_js = get_plotly_js()

    for ccy, cdata in all_countries.items():
        html = generate_country_html(cdata, plotly_js, puma_logo)
        out  = outdir / f"elasticity_foc_{ccy}.html"
        out.write_text(html, encoding="utf-8")
        size_mb = len(html.encode()) / 1024 / 1024
        print(f"  {out.name}  ({size_mb:.1f} MB)")

    idx_html = generate_index_html(all_data, args.top, puma_logo, graas_logo)
    idx_out  = outdir / "elasticity_foc_index.html"
    idx_out.write_text(idx_html, encoding="utf-8")
    print(f"  {idx_out.name}")

    print(f"\nDone. Open: {idx_out.resolve()}")


if __name__ == "__main__":
    main()
