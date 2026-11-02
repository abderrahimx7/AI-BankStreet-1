"""Shared visual components. No data processing or model changes."""
from html import escape

import altair as alt
import pandas as pd
import streamlit as st


GREEN = "#16745F"
BLUE = "#6478B8"


def chart_colors():
    """Palette follows the per-session appearance switch."""
    if st.session_state.get("night_mode", False):
        return {"background": "#152633", "grid": "#2B404D", "text": "#B4C6D2",
                "green": "#67C6A5", "blue": "#99A9E0", "amber": "#E1B37C"}
    return {"background": "#FFFFFF", "grid": "#EDF0F4", "text": "#71818E",
            "green": GREEN, "blue": BLUE, "amber": "#C79359"}


def apply_theme(dark=False):
    css = """
    <style>
    :root { --ink:#182B3C; --muted:#778590; --green:#16745F; }
    html, body, [data-testid="stApp"] { font-family:Inter,Segoe UI,Arial,sans-serif; }
    [data-testid="stAppViewContainer"] { background:#F5F7FA; }
    [data-testid="stHeader"] { background:rgba(245,247,250,.94); }
    .block-container { max-width:1500px; padding:5.25rem 3rem 3rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; }
    h1 { font-weight:650 !important; }
    h3 { font-size:1.12rem !important; font-weight:650 !important; }
    p,li { color:#526373; }
    [data-testid="stCaptionContainer"] p { color:#7A8792; font-size:.78rem; line-height:1.65; }
    [data-testid="stSidebar"] { background:#112331; border-right:1px solid #203746; }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.8rem; }
    [data-testid="stSidebar"] .block-container { padding:1rem; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:#B5C4CD; }
    [data-testid="stSidebar"] hr { border-color:#2A3E4C; }
    [data-testid="stSidebar"] h3 { color:#EAF1F4; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap:.35rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
        border-radius:8px; padding:.7rem .9rem; margin:0; transition:background .15s;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover { background:#1B3342; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background:#244A45; box-shadow:inset 3px 0 #A6D8C1;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p { color:#D8F1E6; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display:none; }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background:#1E3747; border:1px solid #39515F; color:#D8E5EC;
    }
    [data-testid="stMetric"] {
        background:#FFF; border:1px solid #E4E9EE; border-radius:12px;
        padding:1.15rem 1.3rem; min-height:115px; box-shadow:0 2px 3px #152C3C02;
    }
    [data-testid="stMetricLabel"] p { color:#697A89; font-size:.78rem; }
    [data-testid="stMetricValue"] { color:#193446; font-size:1.85rem; letter-spacing:-.055em; font-weight:600; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-color:#E4E9EE !important; border-radius:12px; }
    [data-testid="stExpander"] { background:#FFF; border-color:#E4E9EE; border-radius:10px; }
    [data-testid="stDataFrame"] { border-radius:9px; overflow:hidden; }
    [data-testid="stFileUploader"] { background:#FFF; border:1px dashed #C3D2D9; border-radius:12px; padding:.75rem; }
    [data-testid="stFileUploaderDropzone"] { background:#F5F9F8; border-radius:9px; }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button { border-radius:7px; font-weight:550; }
    [data-testid="stTabs"] [role="tablist"] { gap:1.5rem; border-bottom:1px solid #DFE6EB; }
    [data-testid="stTabs"] [role="tab"] { font-size:.85rem; }
    .bs-brand { display:flex; align-items:center; gap:10px; margin:0 0 2.4rem; }
    .bs-monogram { color:#D8F2E5; background:#2B5A4E; border:1px solid #427564;
        border-radius:9px; width:38px; height:38px; display:grid; place-items:center; font-size:23px; font-weight:700; }
    .bs-brand strong { color:#F1F6F8; font-size:1.2rem; font-weight:600; letter-spacing:-.04em; }
    .bs-brand small { display:block; color:#8BA1AD; font-size:.6rem; letter-spacing:.17em; margin-top:4px; }
    .bs-navlabel { color:#6F8B9B; font-size:.61rem; font-weight:650; letter-spacing:.18em; margin-bottom:.7rem; }
    .bs-side-note { margin-top:2rem; border-top:1px solid #2A3E4C; padding-top:1rem; font-size:.72rem; line-height:1.8; color:#8CA5B3; }
    .bs-side-note b { color:#C7DCD3; font-weight:500; }
    .bs-header { display:flex; justify-content:space-between; align-items:start; gap:20px; margin:.15rem 0 1.75rem; }
    .bs-eyebrow { color:#708492; font-size:.62rem; font-weight:650; letter-spacing:.2em; margin-bottom:.7rem; line-height:1.7; }
    .bs-header h1 { margin:0; padding:0; font-size:2.25rem; line-height:1.18; color:#193143; }
    .bs-header p { margin:.7rem 0 0; font-size:.85rem; max-width:760px; line-height:1.7; color:#71818E; }
    .bs-tag { white-space:nowrap; margin-top:1.7rem; color:#3B7061; background:#EAF4EF; border:1px solid #CEE5D9;
        padding:7px 11px; border-radius:5px; font-size:.64rem; letter-spacing:.05em; }
    .bs-section { display:flex; align-items:center; justify-content:space-between; gap:12px; margin:1.7rem 0 .8rem; }
    .bs-section h3 { padding:0; margin:0; font-size:1.05rem; color:#213B4C; letter-spacing:-.025em; }
    .bs-section span { font-size:.7rem; color:#81909B; }
    .bs-callout { border-left:3px solid #4E9780; background:#ECF4F0; border-radius:0 7px 7px 0;
        padding:.95rem 1.1rem; font-size:.8rem; color:#46665A; line-height:1.75; margin:.8rem 0 1rem; }
    .bs-pill { display:inline-block; font-size:.65rem; color:#657C88; background:#EEF2F5; padding:4px 7px; border-radius:4px; margin-right:5px; }
    .bs-foot { border-top:1px solid #E0E6EC; margin-top:2.5rem; padding-top:1rem; display:flex;
        justify-content:space-between; gap:10px; color:#8D99A4; font-size:.66rem; letter-spacing:.025em; }
    @media(max-width:900px) { .block-container { padding:5rem 1rem 2rem; } .bs-header h1 { font-size:1.8rem; }
        .bs-tag { display:none; } [data-testid="stMetric"] { padding:.9rem; } }
    </style>
    """
    if dark:
        css += """
        <style>
        :root { --ink:#E4EEF4; --muted:#A2B7C5; --green:#67C6A5; color-scheme:dark; }
        [data-testid="stApp"], [data-testid="stAppViewContainer"] { background:#0D1B26; color:#DAE7EF; }
        [data-testid="stHeader"] { background:rgba(13,27,38,.96); color:#D9E7EF; }
        h1,h2,h3,h4,h5,h6 { color:#E4EEF4; }
        p,li,label { color:#B9CAD5; }
        strong { color:#E0EAF0; }
        [data-testid="stCaptionContainer"] p { color:#9CB1BF; }
        [data-testid="stSidebar"] { background:#0A1721; border-right-color:#243C4B; }
        [data-testid="stSidebar"] [data-testid="stToggle"] p { color:#CEE2EC; }
        [data-testid="stMetric"] { background:#152633; border-color:#2B414F; box-shadow:none; }
        [data-testid="stMetricLabel"] p { color:#A7BDCA; }
        [data-testid="stMetricValue"] { color:#ECF4F8; }
        [data-testid="stVerticalBlockBorderWrapper"] > div { border-color:#2B414F !important; background:#152633; }
        [data-testid="stExpander"] { background:#152633; border-color:#2B414F; }
        [data-testid="stExpander"] summary { color:#CEE0EA; }
        [data-testid="stFileUploader"] { background:#152633; border-color:#476657; }
        [data-testid="stFileUploaderDropzone"] { background:#192F3A; color:#C9DEE8; }
        [data-testid="stFileUploaderDropzone"] small { color:#A7BDCA; }
        [data-testid="stFileUploaderFileName"] { color:#DCEAF1; }
        [data-testid="stFileUploaderFile"] { background:#1B3441; color:#DCEAF1; }
        [data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
        [data-testid="stFileUploader"] button { background:#1D3B40; border-color:#41635D; color:#DEF2E9; }
        [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover { background:#28524E; border-color:#67C6A5; }
        [data-testid="stButton"] button p, [data-testid="stDownloadButton"] button p,
        [data-testid="stFileUploader"] button p { color:#DEF2E9; }
        [data-testid="stCheckbox"] label p { color:#C9DCE7; }
        [data-testid="stTabs"] [role="tablist"] { border-bottom-color:#2C424F; }
        [data-testid="stTabs"] [role="tab"] { color:#A7BDCA; }
        [data-testid="stTabs"] [role="tab"] p { color:inherit; }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:#83D7B8; }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color:#67C6A5; }
        [data-testid="stTextInput"] input, [data-testid="stDateInput"] input,
        [data-testid="stNumberInput"] input { background:#1B303E; color:#E0EDF4; }
        [data-testid="stTextInput"] [data-baseweb="input"],
        [data-testid="stDateInput"] [data-baseweb="input"],
        [data-testid="stSelectbox"] [data-baseweb="select"] > div { background:#1B303E; border-color:#38505F; color:#E0EDF4; }
        [data-testid="stWidgetLabel"] p { color:#B9CCD7; }
        [data-baseweb="popover"] [data-baseweb="menu"] { background:#1B303E; color:#E0EDF4; }
        [data-baseweb="popover"] [role="option"] { color:#E0EDF4; }
        [data-baseweb="popover"] [role="option"]:hover { background:#294653; }
        [data-baseweb="calendar"] { background:#1B303E; color:#E0EDF4; }
        [data-baseweb="calendar"] button { color:#E0EDF4; }
        [data-baseweb="calendar"] [aria-selected="true"] { background:#2B6252; color:#F0FFF8; }
        [data-testid="stAlert"] { background:#1C303E; border:1px solid #36505F; color:#CBDDE7; }
        [data-testid="stAlert"] p { color:#CBDDE7; }
        [data-testid="stCode"] { background:#10212D; }
        [data-testid="stCode"] pre, [data-testid="stCode"] code { color:#C9DFE9; }
        [data-testid="stHeader"] button, [data-testid="stToolbar"] button { color:#CDE0EA; }
        /* Streamlit tables are canvas-rendered. Adapt the native light canvas
           as a single unit so headers, cell text and progress bars stay visible. */
        [data-testid="stDataFrame"] { filter:invert(.88) hue-rotate(180deg); border:1px solid #D7C4B5; }
        .bs-header h1 { color:#EBF3F8; }
        .bs-header p { color:#A5BBC9; }
        .bs-eyebrow { color:#9DB7C7; }
        .bs-tag { background:#193B34; border-color:#376352; color:#A2E0C5; }
        .bs-section h3 { color:#E1EDF4; }
        .bs-section span { color:#9BB2C1; }
        .bs-callout { background:#17352F; color:#BBDFCE; border-left-color:#67C6A5; }
        .bs-pill { background:#233C4A; color:#BED4E0; }
        .bs-foot { border-top-color:#293F4E; color:#8CA6B7; }
        hr { border-color:#2A414F; }
        </style>
        """
    st.html(css)


def page_header(eyebrow, title, description, tag="HISTORICAL DATA"):
    st.markdown(
        '<div class="bs-header"><div><div class="bs-eyebrow">'
        + escape(eyebrow) + '</div><h1>' + escape(title)
        + '</h1><p>' + escape(description) + '</p></div><div class="bs-tag">'
        + escape(tag) + '</div></div>', unsafe_allow_html=True,
    )


def section(title, note=""):
    st.markdown(
        '<div class="bs-section"><h3>' + escape(title)
        + '</h3><span>' + escape(note) + '</span></div>',
        unsafe_allow_html=True,
    )


def callout(text):
    st.markdown('<div class="bs-callout">' + escape(text) + '</div>', unsafe_allow_html=True)


def footer():
    st.markdown('<div class="bs-foot"><span>BANKSTREET AI · RESEARCH WORKSPACE</span>'
                '<span>Historical data. Transparent evaluation.</span></div>', unsafe_allow_html=True)


def metric_row(items):
    for column, (label, value) in zip(st.columns(len(items)), items):
        column.metric(label, value)


def line_chart(frame, y_title, height=300, area=False, numeric_index=False):
    """Plot an already calculated, date-indexed frame without changing it."""
    frame = frame.copy()
    frame.index.name = "date"
    values = frame.reset_index().melt("date", var_name="Series", value_name="Value").dropna(subset=["Value"])
    if values.empty:
        st.info("Not enough history to display this series.")
        return
    series = list(frame.columns)
    colors = chart_colors()
    palette = [colors["green"], colors["blue"], colors["amber"], "#B692BD"][:len(series)]
    x_axis = (alt.X("date:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)) if numeric_index else
              alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", tickCount=6, labelAngle=0)))
    x_tooltip = (alt.Tooltip("date:Q", title="Epoch", format="d") if numeric_index else
                 alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"))
    base = alt.Chart(values).encode(
        x=x_axis,
        y=alt.Y("Value:Q", title=y_title, scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", scale=alt.Scale(domain=series, range=palette),
                        legend=alt.Legend(orient="top", title=None, symbolType="stroke")),
        tooltip=[x_tooltip, "Series:N",
                 alt.Tooltip("Value:Q", title=y_title, format=".3f")],
    )
    chart = base.mark_line(strokeWidth=2.2)
    if area and len(series) == 1:
        chart = base.mark_area(opacity=.12) + chart
    chart = chart.properties(height=height).configure(background=colors["background"]).configure_view(stroke=None).configure_axis(
        gridColor=colors["grid"], domain=False, tickSize=0, labelColor=colors["text"], titleColor=colors["text"],
        labelFontSize=10, titleFontSize=10, labelPadding=10,
    ).configure_legend(labelColor=colors["text"], labelFontSize=11, padding=5)
    st.altair_chart(chart, width="stretch", theme=None)
