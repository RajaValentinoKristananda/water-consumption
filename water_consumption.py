import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Water Consumption Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f8fafc; }
    h1, h2, h3 { color: #1e293b; font-weight: 700; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem; background-color: white;
        border-radius: 0.75rem; padding: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; border-radius: 0.5rem;
        padding: 0.75rem 1.5rem; font-weight: 600; color: #64748b; border: none;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0ea5e9; color: white;
    }
    [data-testid="metric-container"] {
        background: white; border-radius: 12px; padding: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 4px solid #0ea5e9;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stSelectbox > div > div { background-color: white; border-radius: 0.5rem; }
    .stMultiSelect > div > div { background-color: white; border-radius: 0.5rem; }
    [data-testid="stSidebar"] { background: white; border-right: 1px solid #e5e7eb; }
    [data-testid="stSidebar"] .stMarkdown h3 { color: #0c4a6e; }
    [data-testid="stFileUploadDropzone"] {
        background: #f0f9ff; border: 2px dashed #0ea5e9; border-radius: 12px;
    }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================
DEFAULT_PRODS = ['PROD 1.3', 'PROD 1.4', 'PROD 1.5', 'PROD 1.6',
                 'PROD 2.5', 'PROD 2.6', 'PROD 2.7']

_COLOR_PALETTE = [
    '#0ea5e9', '#10b981', '#f59e0b', '#ef4444',
    '#8b5cf6', '#ec4899', '#06b6d4', '#f97316',
    '#84cc16', '#6366f1', '#14b8a6', '#e11d48',
    '#a855f7', '#0284c7', '#059669', '#d97706',
]
_UNIT_COLOR_CACHE: dict = {}

def get_unit_color(unit: str) -> str:
    _LEGACY = {
        'PROD 1.3': '#0ea5e9', 'PROD 1.4': '#10b981',
        'PROD 1.5': '#f59e0b', 'PROD 1.6': '#ef4444',
        'PROD 2.5': '#8b5cf6', 'PROD 2.6': '#ec4899',
        'PROD 2.7': '#06b6d4',
    }
    if unit in _LEGACY:
        return _LEGACY[unit]
    if unit not in _UNIT_COLOR_CACHE:
        idx = len(_UNIT_COLOR_CACHE) % len(_COLOR_PALETTE)
        _UNIT_COLOR_CACHE[unit] = _COLOR_PALETTE[idx]
    return _UNIT_COLOR_CACHE[unit]

# ============================================================================
# DATA LOADING
# ============================================================================

def fix_excel_bytes(raw_bytes: bytes) -> bytes:
    EOCD = b'PK\x05\x06'
    idx = raw_bytes.rfind(EOCD)
    if idx != -1:
        return raw_bytes[:idx + 22]
    return raw_bytes


@st.cache_data(show_spinner=False)
def load_raw_data(file_bytes: bytes) -> pd.DataFrame:
    fixed = fix_excel_bytes(file_bytes)
    buf   = io.BytesIO(fixed)

    probe = pd.read_excel(buf, sheet_name=0, header=None, nrows=10)
    header_row = 0
    for i, row in probe.iterrows():
        row_str = ' '.join([str(v).lower() for v in row.values])
        if 'date' in row_str and 'location' in row_str:
            header_row = i
            break

    buf.seek(0)
    df = pd.read_excel(io.BytesIO(fixed), sheet_name=0, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'date' in cl:
            col_map[c] = 'Date'
        elif 'location' in cl:
            col_map[c] = 'Location'
        elif 'pompa' in cl:
            col_map[c] = 'Pompa'
        elif 'water' in cl or 'indicator' in cl:
            col_map[c] = 'Water_Indicator'
    df = df.rename(columns=col_map)

    required = ['Date', 'Location', 'Pompa', 'Water_Indicator']
    for r in required:
        if r not in df.columns:
            st.error(
                f"Column '{r}' not found. "
                f"Please ensure the file contains: Date, Location, Pompa, Water Indicator"
            )
            return pd.DataFrame()

    df['Date']            = pd.to_datetime(df['Date'], errors='coerce')
    df['Water_Indicator'] = pd.to_numeric(df['Water_Indicator'], errors='coerce')
    df = df.dropna(subset=['Date', 'Water_Indicator'])
    df['Pompa'] = df['Pompa'].astype(str).str.strip()
    return df


def process_data(df: pd.DataFrame, selected_prods: list,
                 dedup_method: str, max_spike: float):
    if df.empty or not selected_prods:
        return None

    filt = df[df['Pompa'].isin(selected_prods)].copy()
    if filt.empty:
        return None

    if dedup_method == 'First':
        dedup = filt.groupby(['Date', 'Pompa'])['Water_Indicator'].first().reset_index()
    elif dedup_method == 'Last':
        dedup = filt.groupby(['Date', 'Pompa'])['Water_Indicator'].last().reset_index()
    elif dedup_method == 'Max':
        dedup = filt.groupby(['Date', 'Pompa'])['Water_Indicator'].max().reset_index()
    else:
        dedup = filt.groupby(['Date', 'Pompa'])['Water_Indicator'].mean().reset_index()

    loc_map = filt.groupby('Pompa')['Location'].first().to_dict()
    dedup['Location'] = dedup['Pompa'].map(loc_map)

    pivot = (dedup.pivot(index='Date', columns='Pompa', values='Water_Indicator')
                  .sort_index())

    cons = pivot.diff()
    cons.iloc[0] = 0
    cons = cons.clip(lower=0, upper=max_spike)

    return pivot, cons, dedup, loc_map


# ============================================================================
# HELPERS
# ============================================================================

def fmt_date(d) -> str:
    try:
        return pd.Timestamp(d).strftime('%d %b %Y')
    except Exception:
        return str(d)


def kpi_card_html(prod: str, area: str, total: float, avg: float,
                  max_v: float, pct: float, color: str) -> str:
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return f"""
    <div style="background:white;border-radius:14px;overflow:hidden;
                box-shadow:0 1px 6px rgba(0,0,0,0.08);border-top:4px solid {color};
                padding:16px 18px;font-family:Inter,sans-serif;margin-bottom:4px;">
      <div style="font-size:11px;font-weight:700;color:{color};letter-spacing:1px;
                  text-transform:uppercase;margin-bottom:4px;">{prod}</div>
      <div style="font-size:11px;color:#64748b;margin-bottom:10px;
                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{area}</div>
      <div style="font-size:34px;font-weight:700;color:#0f172a;line-height:1;">{total:,.1f}</div>
      <div style="font-size:12px;color:#94a3b8;margin-bottom:10px;">m³ total</div>
      <div style="display:flex;gap:10px;font-size:11px;margin-bottom:10px;">
        <div style="flex:1;background:#f8fafc;border-radius:8px;padding:6px 8px;">
          <div style="color:#64748b;">Avg / day</div>
          <div style="font-weight:700;color:#0f172a;">{avg:.1f} m³</div>
        </div>
        <div style="flex:1;background:#f8fafc;border-radius:8px;padding:6px 8px;">
          <div style="color:#64748b;">Max / day</div>
          <div style="font-weight:700;color:#0f172a;">{max_v:.1f} m³</div>
        </div>
      </div>
      <div style="height:6px;background:#f0f0f0;border-radius:99px;overflow:hidden;">
        <div style="height:100%;width:{min(pct,100):.1f}%;
                    background:linear-gradient(90deg,rgba({r},{g},{b},0.4),{color});
                    border-radius:99px;"></div>
      </div>
      <div style="font-size:10px;color:#94a3b8;margin-top:4px;">{pct:.1f}% of total</div>
    </div>"""


# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def make_pie(cons_totals: pd.Series, loc_map: dict) -> go.Figure:
    colors = [get_unit_color(p) for p in cons_totals.index]

    def short_area(prod):
        a = loc_map.get(prod, '')
        return a.replace(f'{prod} - ', '').replace(prod, '').strip(' -')

    prods   = cons_totals.index.tolist()
    vals    = cons_totals.values
    areas   = [short_area(p) for p in prods]
    total   = vals.sum() if vals.sum() > 0 else 1
    pcts    = [v / total * 100 for v in vals]

    hover_texts  = [
        f"<b>{p}</b><br>{a}<br>{v:,.1f} m³  ({pct:.1f}%)"
        for p, a, v, pct in zip(prods, areas, vals, pcts)
    ]
    slice_labels = [
        f"<b>{p}</b><br>{a}<br>{v:,.1f} m³<br>{pct:.1f}%"
        for p, a, v, pct in zip(prods, areas, vals, pcts)
    ]

    fig = go.Figure(go.Pie(
        labels=prods,
        values=vals,
        text=slice_labels,
        hovertext=hover_texts,
        hoverinfo='text',
        texttemplate='%{text}',
        textfont=dict(size=10),
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        hole=0.35,
        insidetextorientation='radial',
    ))
    fig.update_layout(
        title=dict(text='Consumption Distribution by Unit', font=dict(size=15, color='#1e293b')),
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom', y=-0.25,
            xanchor='center', x=0.5,
            font=dict(size=10),
            itemsizing='constant',
        ),
        height=480, margin=dict(l=120, r=120, t=50, b=80)
    )
    return fig


def make_bar_total(cons_totals: pd.Series, loc_map: dict) -> go.Figure:
    """For Streamlit display — outside labels, wide margin."""
    prods  = cons_totals.index.tolist()
    vals   = cons_totals.values
    colors = [get_unit_color(p) for p in prods]

    def short_area(prod):
        a = loc_map.get(prod, '')
        return a.replace(f'{prod} - ', '').replace(prod, '').strip(' -')

    areas = [short_area(p) for p in prods]
    y_labels    = [f"{p} — {a}" for p, a in zip(prods, areas)]
    hover_texts = [f"<b>{p}</b><br>{a}<br>{v:,.1f} m³" for p, a, v in zip(prods, areas, vals)]

    fig = go.Figure(go.Bar(
        x=vals,
        y=y_labels,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=1)),
        text=[f"{v:,.0f} m³" for v in vals],
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text',
    ))
    x_max = max(vals) * 1.30 if len(vals) else 1
    fig.update_layout(
        title=dict(text='Total Water Consumption by Unit (m³)', font=dict(size=15, color='#1e293b')),
        paper_bgcolor='white', plot_bgcolor='white',
        height=max(380, len(prods) * 55 + 80),
        margin=dict(l=10, r=20, t=50, b=20),
        xaxis=dict(showgrid=True, gridcolor='#f1f5f9', linecolor='#e2e8f0',
                   tickfont=dict(size=10), range=[0, x_max]),
        yaxis=dict(showgrid=False, linecolor='#e2e8f0',
                   tickfont=dict(size=11), autorange='reversed'),
    )
    return fig


def make_bar_total_html(cons_totals: pd.Series, loc_map: dict) -> go.Figure:
    """For HTML export — robust horizontal bar chart, labels inside bars."""
    prods  = cons_totals.index.tolist()
    # Convert to plain Python floats to avoid numpy serialization issues
    vals   = [float(v) for v in cons_totals.values]
    colors = [get_unit_color(p) for p in prods]

    def short_area(prod):
        a = loc_map.get(prod, '')
        return a.replace(f'{prod} - ', '').replace(prod, '').strip(' -')

    areas       = [short_area(p) for p in prods]
    hover_texts = [f"<b>{p}</b><br>{a}<br>{v:,.1f} m³" for p, a, v in zip(prods, areas, vals)]

    traces = []
    for i, (prod, val, color, area, hover) in enumerate(zip(prods, vals, colors, areas, hover_texts)):
        traces.append(go.Bar(
            x=[val],
            y=[prod],
            orientation='h',
            name=prod,
            showlegend=False,
            marker=dict(color=color, line=dict(color='white', width=1)),
            text=[f"{area} — {val:,.0f} m³"],
            textposition='inside',
            insidetextanchor='start',
            textfont=dict(size=11, color='white'),
            hovertext=[hover],
            hoverinfo='text',
        ))

    fig = go.Figure(data=traces)
    x_max = max(vals) * 1.08 if vals else 1

    fig.update_layout(
        title=dict(text='Total Water Consumption by Unit (m³)', font=dict(size=15, color='#1e293b')),
        paper_bgcolor='white', plot_bgcolor='white',
        barmode='overlay',
        height=max(380, len(prods) * 58 + 80),
        margin=dict(l=90, r=30, t=50, b=20),
        xaxis=dict(
            showgrid=True, gridcolor='#f1f5f9', linecolor='#e2e8f0',
            tickfont=dict(size=10),
            range=[0, x_max],
            autorange=False,
        ),
        yaxis=dict(
            showgrid=False, linecolor='#e2e8f0',
            tickfont=dict(size=12, color='#1e293b'),
            autorange='reversed',
            automargin=True,
        ),
        uniformtext=dict(minsize=8, mode='hide'),
    )
    return fig


def make_line_daily(cons: pd.DataFrame, loc_map: dict, selected_prods: list) -> go.Figure:
    """Dual y-axis linear scale: large units on left axis, small units on right axis."""
    from plotly.subplots import make_subplots

    cols = [p for p in selected_prods if p in cons.columns]
    if not cols:
        return go.Figure()

    x_lbl = [fmt_date(d) for d in cons.index]

    # Classify units: top units (sum >= 10% of max) go left, rest go right
    totals  = {p: float(cons[p].sum()) for p in cols}
    max_tot = max(totals.values()) if totals else 1
    large   = [p for p in cols if totals[p] >= max_tot * 0.10]
    small   = [p for p in cols if totals[p] <  max_tot * 0.10]

    fig = make_subplots(specs=[[{"secondary_y": bool(small)}]])

    for prod in large:
        color = get_unit_color(prod)
        area  = loc_map.get(prod, '')
        fig.add_trace(go.Scatter(
            x=x_lbl,
            y=[float(v) for v in cons[prod].tolist()],
            mode='lines+markers', name=prod,
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            hovertemplate=f'<b>{prod}</b> — {area}<br>%{{x}}<br>%{{y:,.2f}} m³<extra></extra>'
        ), secondary_y=False)

    for prod in small:
        color = get_unit_color(prod)
        area  = loc_map.get(prod, '')
        fig.add_trace(go.Scatter(
            x=x_lbl,
            y=[float(v) for v in cons[prod].tolist()],
            mode='lines+markers', name=f'{prod} ▷',
            line=dict(color=color, width=2, dash='dot'),
            marker=dict(size=5, color=color, symbol='diamond'),
            hovertemplate=f'<b>{prod}</b> — {area}<br>%{{x}}<br>%{{y:,.2f}} m³ (right axis)<extra></extra>'
        ), secondary_y=True)

    title_txt = 'Daily Consumption Trend by Unit (m³/day)'
    if small:
        title_txt += ' — Dual Axis'

    fig.update_layout(
        title=dict(text=title_txt, font=dict(size=15, color='#1e293b')),
        paper_bgcolor='white', plot_bgcolor='white',
        height=450, margin=dict(l=70, r=70, t=60, b=60),
        xaxis=dict(showgrid=False, linecolor='#e2e8f0',
                   tickfont=dict(size=10), tickangle=-30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, font=dict(size=10)),
        hovermode='x unified'
    )
    fig.update_yaxes(
        title_text='m³/day',
        showgrid=True, gridcolor='#f1f5f9',
        rangemode='tozero',
        secondary_y=False
    )
    if small:
        fig.update_yaxes(
            title_text='m³/day (small units ▷)',
            showgrid=False,
            rangemode='tozero',
            secondary_y=True
        )
        fig.add_annotation(
            text='Dashed lines (◆) use right y-axis',
            xref='paper', yref='paper', x=0.01, y=1.08,
            showarrow=False, font=dict(size=10, color='#94a3b8'),
            align='left'
        )
    return fig


def make_stacked_bar(cons: pd.DataFrame, loc_map: dict, selected_prods: list) -> go.Figure:
    fig = go.Figure()
    x_lbl = [fmt_date(d) for d in cons.index]
    for prod in selected_prods:
        if prod not in cons.columns:
            continue
        color = get_unit_color(prod)
        fig.add_trace(go.Bar(
            x=x_lbl, y=cons[prod], name=prod,
            marker_color=color,
            hovertemplate=f'<b>{prod}</b><br>%{{x}}<br>%{{y:,.2f}} m³<extra></extra>'
        ))
    fig.update_layout(
        title=dict(text='Daily Consumption — Stacked (m³)', font=dict(size=15, color='#1e293b')),
        barmode='stack', paper_bgcolor='white', plot_bgcolor='white',
        height=420, margin=dict(l=60, r=20, t=50, b=60),
        xaxis=dict(showgrid=False, linecolor='#e2e8f0', tickfont=dict(size=10), tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9',
                   title=dict(text='m³/day', font=dict(size=12))),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    return fig


def make_heatmap(cons: pd.DataFrame, selected_prods: list) -> go.Figure:
    """FIX: Heatmap was blank because zmin=zmax when one value dominates.
    Use per-row normalization so all units are visible."""
    cols = [p for p in selected_prods if p in cons.columns]
    if not cols:
        return go.Figure()

    z_raw = cons[cols].T.values.astype(float)
    x     = [fmt_date(d) for d in cons.index]

    # Normalize each row (unit) independently so small units are visible
    z_norm = np.zeros_like(z_raw)
    for i in range(len(z_raw)):
        row_max = z_raw[i].max()
        if row_max > 0:
            z_norm[i] = z_raw[i] / row_max * 100
        else:
            z_norm[i] = z_raw[i]

    # Convert to Python lists for reliable JSON serialization in HTML export
    z_norm_list = z_norm.tolist()
    z_raw_list  = z_raw.tolist()

    # Build hover text with actual values
    hover = []
    for i, prod in enumerate(cols):
        row_hover = []
        for j, date in enumerate(x):
            actual = z_raw[i][j]
            row_hover.append(f"<b>{prod}</b><br>{date}<br>{actual:,.2f} m³")
        hover.append(row_hover)

    fig = go.Figure(go.Heatmap(
        z=z_norm_list,
        x=x,
        y=cols,
        customdata=z_raw_list,
        colorscale='Blues',
        zmin=0,
        zmax=100,
        hovertext=hover,
        hoverinfo='text',
        colorbar=dict(
            title=dict(text='Relative Intensity (%)', side='right'),
            ticksuffix='%'
        )
    ))
    fig.update_layout(
        title=dict(text='Daily Consumption Heatmap (relative intensity per unit)', font=dict(size=15, color='#1e293b')),
        paper_bgcolor='white', plot_bgcolor='white',
        height=max(300, len(cols) * 45 + 100),
        margin=dict(l=80, r=80, t=60, b=60),
        xaxis=dict(tickfont=dict(size=9), tickangle=-30),
        yaxis=dict(tickfont=dict(size=11))
    )
    return fig


def make_cumulative(cons: pd.DataFrame, loc_map: dict, selected_prods: list) -> go.Figure:
    """Use log scale so all units are visible. Each trace fills to zero independently."""
    fig = go.Figure()
    x_lbl = [fmt_date(d) for d in cons.index]
    for prod in selected_prods:
        if prod not in cons.columns:
            continue
        color = get_unit_color(prod)
        cumul = cons[prod].cumsum().replace(0, np.nan)
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=x_lbl, y=cumul, mode='lines', name=prod,
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=f'rgba({r},{g},{b},0.08)',
            hovertemplate=f'<b>{prod}</b> cumulative<br>%{{x}}<br>%{{y:,.1f}} m³<extra></extra>'
        ))
    fig.update_layout(
        title=dict(text='Cumulative Consumption (m³) — Log Scale', font=dict(size=15, color='#1e293b')),
        paper_bgcolor='white', plot_bgcolor='white',
        height=420, margin=dict(l=70, r=20, t=50, b=60),
        xaxis=dict(showgrid=False, linecolor='#e2e8f0', tickangle=-30,
                   tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9',
                   type='log',
                   title=dict(text='Cumulative m³ (log scale)', font=dict(size=12))),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    return fig


def make_avg_max_bar(cons: pd.DataFrame, loc_map: dict, selected_prods: list) -> go.Figure:
    def short_area(unit):
        a = loc_map.get(unit, '')
        return a.replace(f'{unit} - ', '').replace(unit, '').strip(' -')

    units  = [u for u in selected_prods if u in cons.columns]
    labels = [f"{u} — {short_area(u)}" for u in units]
    avgs   = [float(cons[u][cons[u] > 0].mean()) if (cons[u] > 0).any() else 0 for u in units]
    maxes  = [float(cons[u][cons[u] > 0].max())  if (cons[u] > 0).any() else 0 for u in units]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Avg / day',
        x=labels, y=avgs,
        marker_color='#0ea5e9',
        text=[f"{v:.1f}" for v in avgs], textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg/day: %{y:,.2f} m³<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        name='Max / day',
        x=labels, y=maxes,
        marker_color='#ef4444',
        text=[f"{v:.1f}" for v in maxes], textposition='outside',
        hovertemplate='<b>%{x}</b><br>Max/day: %{y:,.2f} m³<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='Avg & Max Daily Consumption by Unit (m³/day)', font=dict(size=15, color='#1e293b')),
        barmode='group', paper_bgcolor='white', plot_bgcolor='white',
        height=420, margin=dict(l=60, r=10, t=50, b=100),
        xaxis=dict(showgrid=False, linecolor='#e2e8f0', tickfont=dict(size=10), tickangle=-20),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9',
                   title=dict(text='m³/day', font=dict(size=12))),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


# ============================================================================
# PEAK DAY HELPER  ← NEW
# ============================================================================

def get_peak_day_info(cons_active: pd.DataFrame) -> tuple:
    """Return (peak_value, peak_date_str) for the day with highest total consumption."""
    if cons_active.empty:
        return 0.0, "—"
    daily_totals = cons_active.sum(axis=1)
    peak_val  = float(daily_totals.max())
    peak_date = daily_totals.idxmax()
    peak_str  = pd.Timestamp(peak_date).strftime('%d %b %Y (%A)')
    return peak_val, peak_str


# ============================================================================
# HTML EXPORT
# ============================================================================

def generate_html_report(
    cons_active: pd.DataFrame,
    cons_totals: pd.Series,
    loc_map: dict,
    selected_prods: list,
    period_str: str,
    total_all: float,
    n_days: int,
) -> str:
    import json
    import plotly.io as pio

    def short_area(prod):
        a = loc_map.get(prod, '')
        return a.replace(f'{prod} - ', '').replace(prod, '').strip(' -')

    avg_daily = float(cons_active.sum(axis=1).mean()) if len(cons_active) > 0 else 0
    peak_val, peak_date_str = get_peak_day_info(cons_active)

    # Build chart JSONs
    fig_pie     = make_pie(cons_totals, loc_map)
    fig_bar     = make_bar_total_html(cons_totals, loc_map)   # HTML version: inside labels
    fig_line    = make_line_daily(cons_active, loc_map, selected_prods)
    fig_stacked = make_stacked_bar(cons_active, loc_map, selected_prods)
    fig_heatmap = make_heatmap(cons_active, selected_prods)
    fig_cumul   = make_cumulative(cons_active, loc_map, selected_prods)
    fig_avgmax  = make_avg_max_bar(cons_active, loc_map, selected_prods)

    def fig_json(fig):
        return pio.to_json(fig)

    # KPI summary table rows
    kpi_rows = ""
    for prod in selected_prods:
        if prod not in cons_active.columns:
            continue
        color = get_unit_color(prod)
        d     = cons_active[prod]
        d_pos = d[d > 0]
        total_v = round(d.sum(), 1)
        avg_v   = round(d_pos.mean(), 2) if len(d_pos) else 0
        max_v   = round(d_pos.max(),  1) if len(d_pos) else 0
        pct_v   = round(total_v / total_all * 100, 1) if total_all > 0 else 0
        area    = short_area(prod)
        kpi_rows += f"""
        <tr>
          <td><span style="display:inline-block;width:10px;height:10px;
              border-radius:50%;background:{color};margin-right:6px;"></span>
              <b>{prod}</b></td>
          <td>{area}</td>
          <td>{total_v:,.1f}</td>
          <td>{avg_v:,.2f}</td>
          <td>{max_v:,.1f}</td>
          <td>{pct_v:.1f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Water Consumption Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:Inter,sans-serif;background:#f8fafc;color:#1e293b}}
  .header{{background:white;border-bottom:1px solid #e5e7eb;padding:18px 32px;
            display:flex;align-items:center;gap:16px;}}
  .logo{{width:48px;height:48px;background:linear-gradient(135deg,#0ea5e9,#0284c7);
          border-radius:12px;display:flex;align-items:center;justify-content:center;
          font-size:24px;flex-shrink:0;}}
  .header-title{{font-size:24px;font-weight:800;color:#0c4a6e}}
  .header-sub{{font-size:13px;color:#94a3b8;margin-top:3px}}
  .badge{{background:linear-gradient(90deg,#0ea5e9,#0284c7);color:white;
           padding:8px 18px;border-radius:20px;font-size:13px;font-weight:600;
           display:inline-block;margin:16px 32px 0 auto;}}
  .container{{max-width:1400px;margin:0 auto;padding:24px 32px}}
  .section-title{{font-size:20px;font-weight:700;color:#0c4a6e;margin:28px 0 16px;
                   border-left:4px solid #0ea5e9;padding-left:12px;}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}}
  .kpi-card{{background:white;border-radius:12px;padding:20px;
              box-shadow:0 1px 4px rgba(0,0,0,0.08);border-left:4px solid #0ea5e9;}}
  .kpi-label{{font-size:12px;color:#64748b;font-weight:600;text-transform:uppercase;
               letter-spacing:.5px;margin-bottom:6px}}
  .kpi-value{{font-size:28px;font-weight:800;color:#0f172a}}
  .kpi-unit{{font-size:13px;color:#94a3b8;margin-top:2px}}
  .kpi-sub{{font-size:11px;color:#ef4444;font-weight:600;margin-top:4px}}
  .chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
  .chart-full{{margin-bottom:20px}}
  .chart-card{{background:white;border-radius:14px;padding:4px;
                box-shadow:0 1px 6px rgba(0,0,0,0.07);}}
  table{{width:100%;border-collapse:collapse;background:white;border-radius:12px;
         overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.08)}}
  thead th{{background:#0c4a6e;color:white;padding:12px 14px;text-align:left;
             font-size:12px;font-weight:700;letter-spacing:.5px;text-transform:uppercase}}
  tbody tr:nth-child(even){{background:#f0f9ff}}
  tbody td{{padding:10px 14px;font-size:13px;border-bottom:1px solid #f1f5f9}}
  .footer{{text-align:center;color:#94a3b8;padding:24px;font-size:12px;margin-top:32px;
            background:white;border-radius:12px}}
  @media(max-width:768px){{
    .kpi-grid{{grid-template-columns:1fr 1fr}}
    .chart-grid{{grid-template-columns:1fr}}
  }}
</style>
</head>
<body>

<div class="header">
  <div class="logo">💧</div>
  <div>
    <div class="header-title">Water Consumption Dashboard</div>
    <div class="header-sub">Daily water consumption analysis from Water Meter Raw Data — Plant 1</div>
  </div>
  <div class="badge">📅 {period_str} &nbsp;|&nbsp; {n_days} days</div>
</div>

<div class="container">

  <div class="section-title">Summary KPIs</div>
  <div class="kpi-grid">
    <div class="kpi-card" style="border-color:#0ea5e9">
      <div class="kpi-label">💧 Total Consumption</div>
      <div class="kpi-value">{total_all:,.1f}</div>
      <div class="kpi-unit">m³</div>
    </div>
    <div class="kpi-card" style="border-color:#10b981">
      <div class="kpi-label">📅 Days of Data</div>
      <div class="kpi-value">{n_days}</div>
      <div class="kpi-unit">days</div>
    </div>
    <div class="kpi-card" style="border-color:#f59e0b">
      <div class="kpi-label">📊 Average / Day</div>
      <div class="kpi-value">{avg_daily:,.1f}</div>
      <div class="kpi-unit">m³/day</div>
    </div>
    <div class="kpi-card" style="border-color:#ef4444">
      <div class="kpi-label">🔺 Peak Day</div>
      <div class="kpi-value">{peak_val:,.1f}</div>
      <div class="kpi-unit">m³</div>
      <div class="kpi-sub">📅 {peak_date_str}</div>
    </div>
  </div>

  <div class="section-title">Unit Statistics</div>
  <table>
    <thead>
      <tr>
        <th>Unit</th><th>Area</th><th>Total (m³)</th>
        <th>Avg/day (m³)</th><th>Max/day (m³)</th><th>% of Total</th>
      </tr>
    </thead>
    <tbody>{kpi_rows}</tbody>
  </table>

  <div class="section-title">Visualisations</div>
  <div class="chart-grid">
    <div class="chart-card"><div id="pie"></div></div>
    <div class="chart-card"><div id="bar"></div></div>
  </div>
  <div class="chart-card chart-full" style="margin-bottom:20px"><div id="cumul"></div></div>
  <div class="chart-card chart-full" style="margin-bottom:20px"><div id="avgmax"></div></div>

  <div class="section-title">Daily Trends</div>
  <div class="chart-card chart-full"><div id="line"></div></div>
  <div class="chart-card chart-full" style="margin-top:20px"><div id="stacked"></div></div>
  <div class="chart-card chart-full" style="margin-top:20px"><div id="heat"></div></div>

  <div class="footer">
    <b>Water Consumption Dashboard</b> &nbsp;|&nbsp;
    PT Güntner Indonesia &nbsp;|&nbsp; Water Monitoring System<br>
    <span style="font-size:11px;">Generated on {pd.Timestamp.now().strftime('%d %B %Y %H:%M')}</span>
  </div>
</div>

<script>
var cfg = {{responsive:true, displayModeBar:true,
            modeBarButtonsToRemove:['lasso2d','select2d']}};

// Store each figure JSON once, then access .data and .layout separately
var _pie     = {fig_json(fig_pie)};
var _bar     = {fig_json(fig_bar)};
var _line    = {fig_json(fig_line)};
var _stacked = {fig_json(fig_stacked)};
var _heat    = {fig_json(fig_heatmap)};
var _cumul   = {fig_json(fig_cumul)};
var _avgmax  = {fig_json(fig_avgmax)};

Plotly.newPlot('pie',     _pie.data,     _pie.layout,     cfg);
Plotly.newPlot('bar',     _bar.data,     _bar.layout,     cfg);
Plotly.newPlot('line',    _line.data,    _line.layout,    cfg);
Plotly.newPlot('stacked', _stacked.data, _stacked.layout, cfg);
Plotly.newPlot('heat',    _heat.data,    _heat.layout,    cfg);
Plotly.newPlot('cumul',   _cumul.data,   _cumul.layout,   cfg);
Plotly.newPlot('avgmax',  _avgmax.data,  _avgmax.layout,  cfg);
</script>
</body>
</html>"""
    return html



st.markdown("""
<div style="background:white;border-bottom:1px solid #e5e7eb;padding:14px 28px;
            display:flex;align-items:center;gap:16px;margin:-4rem -4rem 1.5rem -4rem;">
  <div style="width:48px;height:48px;background:linear-gradient(135deg,#0ea5e9,#0284c7);
              border-radius:12px;display:flex;align-items:center;justify-content:center;
              font-size:24px;flex-shrink:0;">💧</div>
  <div>
    <div style="font-size:26px;font-weight:800;color:#0c4a6e;line-height:1.2;">
      Water Consumption Dashboard</div>
    <div style="font-size:14px;color:#94a3b8;margin-top:2px;">
      Daily water consumption analysis from Water Meter Raw Data — Plant 1</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 📁 Upload Data")
    uploaded = st.file_uploader(
        "Upload waterawdata file (*.xlsx)",
        type=['xlsx'],
        help="Upload waterawdata_GI.xlsx or any file with the same format"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    if uploaded:
        raw_bytes = uploaded.read()
        with st.spinner("Reading data..."):
            raw_df = load_raw_data(raw_bytes)

        if not raw_df.empty:
            available_prods = sorted(raw_df['Pompa'].unique().tolist())
            default_sel     = [p for p in DEFAULT_PRODS if p in available_prods]
            if not default_sel:
                default_sel = available_prods[:min(7, len(available_prods))]

            selected_prods = st.multiselect(
                "Select Units",
                options=available_prods,
                default=default_sel,
                help="Select one or more units to analyse"
            )

            st.markdown("---")
            st.markdown("### 📅 Date Filter")
            min_date = raw_df['Date'].min().date()
            max_date = raw_df['Date'].max().date()

            date_from = st.date_input("From date", value=min_date,
                                      min_value=min_date)
            date_to   = st.date_input("To date",   value=max_date,
                                      min_value=min_date)

            st.markdown("---")
            st.markdown("### 🔧 Calculation Options")

            dedup_method = st.selectbox(
                "Deduplication Method",
                ['First', 'Last', 'Max', 'Mean'],
                index=0,
                help=(
                    "When multiple readings exist for the same unit "
                    "on the same day, use this value"
                )
            )

            max_spike = st.number_input(
                "Max daily consumption limit (m³)",
                min_value=10.0, max_value=50000.0,
                value=5000.0, step=100.0,
                help="Values above this threshold will be clipped (likely data errors)"
            )

            st.markdown("---")
            st.markdown("### 📊 Display Charts")
            show_pie     = st.checkbox("Pie Chart — Distribution",      value=True)
            show_bar     = st.checkbox("Bar Chart — Total",              value=True)
            show_line    = st.checkbox("Line Chart — Daily Trend",       value=True)
            show_stacked = st.checkbox("Stacked Bar — Daily",            value=True)
            show_heatmap = st.checkbox("Heatmap",                        value=True)
            show_cumul   = st.checkbox("Cumulative Consumption",         value=True)
            show_box     = st.checkbox("Avg & Max Bar Chart",            value=True)
        else:
            selected_prods = []
    else:
        raw_df         = pd.DataFrame()
        selected_prods = DEFAULT_PRODS
        dedup_method   = 'First'
        max_spike      = 5000.0
        date_from      = None
        date_to        = None
        show_pie = show_bar = show_line = show_stacked = True
        show_heatmap = show_cumul = show_box = True

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#94a3b8;text-align:center;'>"
        "PT Güntner Indonesia<br>Water Monitoring System</div>",
        unsafe_allow_html=True
    )


# ============================================================================
# MAIN CONTENT
# ============================================================================
if uploaded is None or raw_df.empty:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
      <div style="font-size:72px;margin-bottom:20px;">💧</div>
      <div style="font-size:24px;font-weight:700;color:#0c4a6e;margin-bottom:12px;">
        Upload Your Water Meter Data File</div>
      <div style="font-size:15px;color:#64748b;max-width:500px;margin:0 auto 32px;">
        Upload <b>waterawdata_GI.xlsx</b> or a file with the same format via the sidebar
        to start analysing water consumption.
      </div>
      <div style="display:flex;justify-content:center;gap:24px;flex-wrap:wrap;">
        <div style="background:#f0f9ff;border-radius:12px;padding:20px 28px;min-width:160px;">
          <div style="font-size:28px;margin-bottom:8px;">📊</div>
          <div style="font-size:13px;font-weight:600;color:#0c4a6e;">7 Chart Types</div>
        </div>
        <div style="background:#f0fdf4;border-radius:12px;padding:20px 28px;min-width:160px;">
          <div style="font-size:28px;margin-bottom:8px;">⚙️</div>
          <div style="font-size:13px;font-weight:600;color:#14532d;">Flexible Filters</div>
        </div>
        <div style="background:#fff7ed;border-radius:12px;padding:20px 28px;min-width:160px;">
          <div style="font-size:28px;margin-bottom:8px;">📥</div>
          <div style="font-size:13px;font-weight:600;color:#7c2d12;">CSV Export</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# Apply date filter
if date_from and date_to:
    raw_df = raw_df[(raw_df['Date'].dt.date >= date_from) &
                    (raw_df['Date'].dt.date <= date_to)]

if not selected_prods:
    st.warning("⚠️ Please select at least one unit from the sidebar.")
    st.stop()

with st.spinner("Processing data..."):
    result = process_data(raw_df, selected_prods, dedup_method, max_spike)

if not result:
    st.error("Failed to process data. Please check the file and your unit selection.")
    st.stop()

pivot, cons, dedup_df, loc_map = result

cons_active = cons.iloc[1:]
cons_totals = cons_active.sum()
total_all   = cons_totals.sum()

period_str = (f"{fmt_date(cons.index.min())}  →  {fmt_date(cons.index.max())}"
              if len(cons) > 0 else "—")
n_days = len(cons)

# Get peak day info
peak_val, peak_date_str = get_peak_day_info(cons_active)

st.markdown(
    f'<div style="display:flex;justify-content:flex-end;margin-bottom:16px;">'
    f'<div style="background:linear-gradient(90deg,#0ea5e9,#0284c7);color:white;'
    f'padding:10px 22px;border-radius:24px;font-size:14px;font-weight:600;'
    f'box-shadow:0 3px 10px rgba(14,165,233,0.35);">'
    f'📅 {period_str} &nbsp;|&nbsp; {n_days} days of data</div></div>',
    unsafe_allow_html=True
)

# ============================================================================
# TABS
# ============================================================================
tab_ov, tab_daily, tab_table, tab_raw = st.tabs([
    "📊 Overview", "📈 Daily Analysis", "📋 Data Table", "🗃️ Raw Data"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_ov:
    st.markdown("## Performance by Unit")

    cols_kpi = st.columns(len(selected_prods))
    for i, prod in enumerate(selected_prods):
        if prod not in cons.columns:
            continue
        area       = loc_map.get(prod, '')
        area_short = area.replace(f'{prod} - ', '').replace(prod, '').strip(' -')
        total_v    = float(cons_totals.get(prod, 0))
        d_vals     = cons_active[prod][cons_active[prod] > 0] if prod in cons_active else pd.Series()
        avg_v      = float(d_vals.mean()) if len(d_vals) > 0 else 0
        max_v      = float(d_vals.max())  if len(d_vals) > 0 else 0
        pct_v      = total_v / total_all * 100 if total_all > 0 else 0
        color      = get_unit_color(prod)
        with cols_kpi[i]:
            st.markdown(
                kpi_card_html(prod, area_short, total_v, avg_v, max_v, pct_v, color),
                unsafe_allow_html=True
            )

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💧 Total Consumption", f"{total_all:,.1f} m³")
    m2.metric("📅 Days of Data",       f"{n_days} days")
    avg_daily_total = float(cons_active.sum(axis=1).mean()) if len(cons_active) > 0 else 0
    m3.metric("📊 Average / Day",     f"{avg_daily_total:,.1f} m³")
    # FIX: Show peak day with date
    m4.metric("🔺 Peak Day", f"{peak_val:,.1f} m³", delta=peak_date_str)

    st.markdown("---")
    st.markdown("## Visualisations")

    if show_pie or show_bar:
        c1, c2 = st.columns(2)
        if show_pie:
            with c1:
                st.plotly_chart(make_pie(cons_totals, loc_map), use_container_width=True)
        if show_bar:
            with c2:
                st.plotly_chart(make_bar_total(cons_totals, loc_map), use_container_width=True)

    if show_cumul:
        st.plotly_chart(
            make_cumulative(cons_active, loc_map, selected_prods),
            use_container_width=True
        )
    if show_box:
        st.plotly_chart(
            make_avg_max_bar(cons_active, loc_map, selected_prods),
            use_container_width=True
        )

    # ── HTML Export ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Export Report")
    st.info(
        "Download a **standalone HTML report** — fully interactive charts, "
        "works offline without any internet connection or Python installation."
    )
    with st.spinner("Building HTML report..."):
        html_report = generate_html_report(
            cons_active, cons_totals, loc_map, selected_prods,
            period_str, total_all, n_days
        )
    fname = f"water_consumption_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.html"
    st.download_button(
        label="⬇️ Download Full Report (HTML)",
        data=html_report.encode('utf-8'),
        file_name=fname,
        mime='text/html',
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – DAILY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_daily:
    st.markdown("## Daily Consumption Trends")

    if show_line:
        st.plotly_chart(
            make_line_daily(cons_active, loc_map, selected_prods),
            use_container_width=True
        )

    if show_stacked:
        st.plotly_chart(
            make_stacked_bar(cons_active, loc_map, selected_prods),
            use_container_width=True
        )

    if show_heatmap:
        st.markdown("### Daily Intensity Heatmap")
        st.plotly_chart(make_heatmap(cons_active, selected_prods), use_container_width=True)

    st.markdown("---")
    st.markdown("### Statistics by Unit")
    stats_rows = []
    for prod in selected_prods:
        if prod not in cons_active.columns:
            continue
        d     = cons_active[prod]
        d_pos = d[d > 0]
        stats_rows.append({
            'PROD':         prod,
            'Area':         loc_map.get(prod, '').replace(f'{prod} - ', ''),
            'Total (m³)':   round(d.sum(), 1),
            'Avg/day (m³)': round(d_pos.mean(), 2) if len(d_pos) else 0,
            'Max/day (m³)': round(d_pos.max(),  1) if len(d_pos) else 0,
            'Min/day (m³)': round(d_pos.min(),  1) if len(d_pos) else 0,
            'Std Dev':      round(d_pos.std(),  2) if len(d_pos) else 0,
            '% of Total':   round(d.sum() / total_all * 100, 1) if total_all > 0 else 0,
        })

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        def style_stats(row):
            color = get_unit_color(row['PROD'])
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            return [f'background-color:rgba({r},{g},{b},0.07)'] * len(row)

        styled = (stats_df.style
                  .apply(style_stats, axis=1)
                  .format({c: '{:,.1f}' for c in stats_df.columns
                           if stats_df[c].dtype in [float, 'float64']})
                  .set_properties(**{'text-align': 'center',
                                     'border': '1px solid #e2e8f0',
                                     'padding': '8px'})
                  .set_table_styles([{
                      'selector': 'thead th',
                      'props': [('background-color', '#0c4a6e'), ('color', 'white'),
                                ('font-weight', 'bold'), ('text-align', 'center'),
                                ('padding', '10px')]
                  }]))
        st.dataframe(styled, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – DATA TABLE
# ─────────────────────────────────────────────────────────────────────────────
with tab_table:
    st.markdown("## Daily Consumption Table (m³)")
    st.info(
        "Consumption = Today's Water Indicator − Previous Day's Water Indicator. "
        "First row = 0 (no prior data available)."
    )

    disp = cons.copy()
    disp.index = [fmt_date(d) for d in disp.index]
    disp.index.name = 'Date'
    active_cols = [p for p in selected_prods if p in disp.columns]
    disp = disp[active_cols]
    disp['TOTAL'] = disp.sum(axis=1)

    def short_area(unit):
        a = loc_map.get(unit, '')
        return a.replace(f'{unit} - ', '').replace(unit, '').strip(' -')

    loc_row = pd.DataFrame(
        {p: short_area(p) for p in active_cols} | {'TOTAL': ''},
        index=['📍 Location']
    )
    total_row = pd.DataFrame(disp.sum()).T
    total_row.index = ['TOTAL']
    disp_full = pd.concat([loc_row, disp, total_row])

    def highlight_special(row):
        if row.name == 'TOTAL':
            return ['background-color:#0c4a6e;color:white;font-weight:bold'] * len(row)
        if row.name == '📍 Location':
            return ['background-color:#e0f2fe;color:#0c4a6e;font-style:italic;font-weight:600'] * len(row)
        all_idx = list(disp_full.index)
        idx = all_idx.index(row.name)
        return [('background-color:#f0f9ff' if idx % 2 == 0 else 'background-color:white')] * len(row)

    def fmt_cell(v):
        try:
            return f'{float(v):,.1f}'
        except (ValueError, TypeError):
            return str(v)

    styled_tbl = (disp_full.style
                  .apply(highlight_special, axis=1)
                  .format(fmt_cell)
                  .set_properties(**{'text-align': 'center',
                                     'border': '1px solid #e2e8f0',
                                     'padding': '7px'})
                  .set_table_styles([{
                      'selector': 'thead th',
                      'props': [('background-color', '#0284c7'), ('color', 'white'),
                                ('font-weight', 'bold'), ('text-align', 'center'),
                                ('padding', '10px'), ('border', '1px solid #0284c7')]
                  }]))

    st.dataframe(styled_tbl, use_container_width=True, height=520)

    c1, c2 = st.columns(2)
    with c1:
        csv_cons = disp_full.to_csv().encode('utf-8')
        st.download_button(
            "⬇️ Download Daily Consumption Table (CSV)",
            csv_cons, 'water_consumption_daily.csv', 'text/csv',
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### Monthly Summary")
    cons_monthly_data = []
    tmp   = cons.copy()
    tmp.index = pd.to_datetime([d for d in tmp.index])
    tmp_m = tmp.resample('ME').sum()
    for d, row in tmp_m.iterrows():
        r = {'Month': d.strftime('%B %Y')}
        for prod in selected_prods:
            if prod in row.index:
                r[prod] = round(float(row[prod]), 1)
        r['TOTAL'] = round(sum(row[p] for p in selected_prods if p in row.index), 1)
        cons_monthly_data.append(r)

    if cons_monthly_data:
        monthly_df = pd.DataFrame(cons_monthly_data)
        st.dataframe(monthly_df, use_container_width=True, hide_index=True)

        with c2:
            csv_monthly = monthly_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Monthly Summary (CSV)",
                csv_monthly, 'water_consumption_monthly.csv', 'text/csv',
                use_container_width=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
with tab_raw:
    st.markdown("## Raw Data (after Deduplication)")
    st.info(
        f"Showing data after deduplication using method: **{dedup_method}**. "
        f"Total rows: {len(dedup_df):,}"
    )

    show_prods_raw = st.multiselect(
        "Filter Units", options=selected_prods,
        default=selected_prods[:3], key='raw_prod_filter'
    )

    raw_view = (dedup_df[dedup_df['Pompa'].isin(show_prods_raw)].copy()
                if show_prods_raw else dedup_df.copy())
    raw_view['Date'] = raw_view['Date'].dt.strftime('%d-%b-%Y')
    raw_view = raw_view.rename(columns={'Water_Indicator': 'Water Indicator (m³)'})
    raw_view['Water Indicator (m³)'] = raw_view['Water Indicator (m³)'].round(2)

    st.dataframe(
        raw_view[['Date', 'Location', 'Pompa', 'Water Indicator (m³)']].reset_index(drop=True),
        use_container_width=True, height=500, hide_index=True
    )

    csv_raw = raw_view.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Raw Data (CSV)",
        csv_raw, 'water_raw_dedup.csv', 'text/csv'
    )

    st.markdown("---")
    st.markdown("### Pivot Table — Daily Water Indicator")
    pivot_disp = pivot.copy()
    pivot_disp.index = [fmt_date(d) for d in pivot_disp.index]
    pivot_disp = pivot_disp[[p for p in selected_prods if p in pivot_disp.columns]]
    st.dataframe(pivot_disp.round(1), use_container_width=True, height=400)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#94a3b8;padding:1.5rem;
            background:white;border-radius:0.75rem;font-size:13px;'>
  <b style='color:#0c4a6e;'>Water Consumption Dashboard</b> &nbsp;|&nbsp;
  PT Güntner Indonesia &nbsp;|&nbsp; Water Monitoring System
</div>
""", unsafe_allow_html=True)