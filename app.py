# advanced_water_dashboard.py
#
# National Water Resources Intelligence Dashboard (Version 12.3 - Robust State Management)
# This version fully adopts st.session_state for API key management, mirroring best
# practices and fixing all AI-related state errors.

# --- IMPORTANT SETUP INSTRUCTION ---
# For the custom theme to work, you MUST create a folder named '.streamlit' in the
# same directory as this script. Inside that folder, create a file named 'config.toml'
# and paste the content from the provided config.toml file into it.

# --- 0. CORE LIBRARY IMPORTS ---
# Standard Libraries
import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import warnings
import time
from datetime import datetime
import base64
from contextlib import contextmanager

# Visualization Libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import altair as alt # For animated, declarative charts
from streamlit_lottie import st_lottie # For loading animations

# AI and Machine Learning Libraries
import google.generativeai as genai
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- 1. APPLICATION WIDE CONFIGURATION & STYLING ---
st.set_page_config(
    layout='wide',
    page_title='AquaSphere-National Water Intelligence Dashboard',
    page_icon="💧"
)
warnings.filterwarnings("ignore")

# Custom CSS Injection for modern UI/UX
st.markdown("""

""", unsafe_allow_html=True)


def set_page_background(svg_content):
    """
    Sets a base64 encoded SVG as the page background.
    """
    b64 = base64.b64encode(svg_content.encode('utf-8')).decode("utf-8")
    page_bg_css = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:image/svg+xml;base64,{b64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"], [data-testid="stSidebar"] {{
        background: rgba(0, 0, 0, 0);
    }}
    </style>
    """
    st.markdown(page_bg_css, unsafe_allow_html=True)

# A simple SVG for a subtle background pattern
svg_background = """
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 800 800">
    <g fill-opacity="0.03">
        <circle fill="#7792E3" cx="400" cy="400" r="600"/>
        <circle fill="#262730" cx="400" cy="400" r="500"/>
        <circle fill="#0E1117" cx="400" cy="400" r="400"/>
        <circle fill="#262730" cx="400" cy="400" r="300"/>
        <circle fill="#7792E3" cx="400" cy="400" r="200"/>
        <circle fill="#0E1117" cx="400" cy="400" r="100"/>
    </g>
</svg>
"""
set_page_background(svg_background)

# --- 2. ADVANCED HELPER & ANALYTICS FUNCTIONS ---

@contextmanager
def card(title=""):
    """A context manager to wrap Streamlit elements in a styled card div."""
    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
    if title:
        st.subheader(title)
    yield
    st.markdown("</div>", unsafe_allow_html=True)


# FIX: Embedded Lottie animation data to remove external dependency
LOTTIE_ANIMATION_DATA = {
    "v": "5.5.7", "fr": 30, "ip": 0, "op": 150, "w": 512, "h": 512, "nm": "Water Drop", "ddd": 0, "assets": [],
    "layers": [
        {"ddd": 0, "ind": 1, "ty": 4, "nm": "Drop 2", "sr": 1, "ks": {
            "o": {"a": 0, "k": 100, "ix": 11}, "r": {"a": 0, "k": 0, "ix": 10},
            "p": {"a": 1, "k": [
                {"i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}, "t": 30, "s": [256, -100, 0]},
                {"i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}, "t": 60, "s": [256, 256, 0]},
                {"i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}, "t": 90, "s": [256, 624, 0]},
                {"t": 120, "s": [256, 256, 0]}
            ], "ix": 2},
            "a": {"a": 0, "k": [0, 0, 0], "ix": 1}, "s": {"a": 0, "k": [100, 100, 100], "ix": 6}
        }, "ao": 0, "shapes": [
            {"ty": "gr", "it": [
                {"ind": 0, "ty": "sh", "ix": 1, "ks": {"a": 0, "k": {
                    "i": [[-44.184, 0], [0, 80.667], [44.184, 0], [0, -80.667]],
                    "o": [[44.184, 0], [0, -80.667], [-44.184, 0], [0, 80.667]],
                    "v": [[0, -146], [-80.667, 0], [0, 146], [80.667, 0]], "c": True
                }, "ix": 2}, "nm": "Path 1", "mn": "ADBE Vector Shape - Group", "hd": False},
                {"ty": "fill", "c": {"a": 0, "k": [0.125490196078, 0.541176470588, 0.835294117647, 1], "ix": 4}, "o": {"a": 0, "k": 100, "ix": 5}, "nm": "Fill 1", "mn": "ADBE Vector Graphic - Fill", "hd": False}
            ], "nm": "Shape Group", "np": 2, "cix": 2, "bm": 0, "ix": 1, "mn": "ADBE Vector Group", "hd": False}
        ], "ip": 0, "op": 150, "st": 0, "bm": 0},
        {"ddd": 0, "ind": 2, "ty": 4, "nm": "Drop 1", "sr": 1, "ks": {
            "o": {"a": 0, "k": 100, "ix": 11}, "r": {"a": 0, "k": 0, "ix": 10},
            "p": {"a": 1, "k": [
                {"i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}, "t": 0, "s": [256, -100, 0]},
                {"i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}, "t": 30, "s": [256, 256, 0]},
                {"i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}, "t": 60, "s": [256, 624, 0]},
                {"t": 90, "s": [256, 256, 0]}
            ], "ix": 2},
            "a": {"a": 0, "k": [0, 0, 0], "ix": 1}, "s": {"a": 0, "k": [100, 100, 100], "ix": 6}
        }, "ao": 0, "shapes": [
            {"ty": "gr", "it": [
                {"ind": 0, "ty": "sh", "ix": 1, "ks": {"a": 0, "k": {
                    "i": [[-44.184, 0], [0, 80.667], [44.184, 0], [0, -80.667]],
                    "o": [[44.184, 0], [0, -80.667], [-44.184, 0], [0, 80.667]],
                    "v": [[0, -146], [-80.667, 0], [0, 146], [80.667, 0]], "c": True
                }, "ix": 2}, "nm": "Path 1", "mn": "ADBE Vector Shape - Group", "hd": False},
                {"ty": "fill", "c": {"a": 0, "k": [0.247058823529, 0.650980392157, 0.901960784314, 1], "ix": 4}, "o": {"a": 0, "k": 100, "ix": 5}, "nm": "Fill 1", "mn": "ADBE Vector Graphic - Fill", "hd": False}
            ], "nm": "Shape Group", "np": 2, "cix": 2, "bm": 0, "ix": 1, "mn": "ADBE Vector Group", "hd": False}
        ], "ip": 0, "op": 150, "st": 0, "bm": 0},
        {"ddd": 0, "ind": 3, "ty": 4, "nm": "Puddle", "sr": 1, "ks": {
            "o": {"a": 0, "k": 100, "ix": 11}, "r": {"a": 0, "k": 0, "ix": 10},
            "p": {"a": 0, "k": [256, 452, 0], "ix": 2},
            "a": {"a": 0, "k": [292.5, 75.5, 0], "ix": 1},
            "s": {"a": 1, "k": [
                {"i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}, "t": 30, "s": [0, 0, 100]},
                {"i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}, "t": 60, "s": [57, 57, 100]},
                {"i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}, "t": 90, "s": [30, 30, 100]},
                {"t": 120, "s": [57, 57, 100]}
            ], "ix": 6}
        }, "ao": 0, "shapes": [
            {"ty": "gr", "it": [
                {"ind": 0, "ty": "sh", "ix": 1, "ks": {"a": 0, "k": {
                    "i": [[-159.988, 0], [0, 41.694], [159.988, 0], [0, -41.694]],
                    "o": [[159.988, 0], [0, -41.694], [-159.988, 0], [0, 41.694]],
                    "v": [[0, -75.5], [-292, 0], [0, 75.5], [292, 0]], "c": True
                }, "ix": 2}, "nm": "Path 1", "mn": "ADBE Vector Shape - Group", "hd": False},
                {"ty": "fill", "c": {"a": 0, "k": [0.247, 0.651, 0.902, 1], "ix": 4}, "o": {"a": 0, "k": 100, "ix": 5}, "nm": "Fill 1", "mn": "ADBE Vector Graphic - Fill", "hd": False}
            ], "nm": "Shape Group", "np": 2, "cix": 2, "bm": 0, "ix": 1, "mn": "ADBE Vector Group", "hd": False}
        ], "ip": 0, "op": 150, "st": 0, "bm": 0}
    ]
}

def animated_metric(column, label, value, unit="", help_text="", delta=None, delta_color="normal"):
    """Displays an st.metric with a number-counting animation."""
    placeholder = column.empty()
    is_float = isinstance(value, float)
    num_decimals = 2 if is_float else 0
    steps = 50
    sleep_duration = 0.8 / (steps + 1)
    for i in range(steps + 1):
        current_display_value = (value / steps) * i
        with placeholder.container():
             st.metric(
                label=label,
                value=f"{current_display_value:,.{num_decimals}f} {unit}",
                delta=delta,
                delta_color=delta_color,
                help=help_text
            )
        time.sleep(sleep_duration)

def get_gemini_response(prompt: str) -> str | None:
    """Centralized function to communicate with the Gemini API. Assumes API is already configured."""
    if st.session_state.get('ai_disabled', False) or not st.session_state.get("gemini_key"):
        st.error("AI features are disabled or API key is not provided.")
        return None
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        with st.spinner("🤖 Communicating with Google Gemini AI..."):
            response = model.generate_content(prompt, request_options={"timeout": 180})
        return response.text
    except Exception as e:
        st.error(f"Error with Gemini API: {e}")
        return None

@st.cache_data
def perform_ai_column_mapping(raw_columns: list, target_schema: list, file_name: str) -> dict | None:
    """Uses AI to map raw CSV columns to the application's standard schema."""
    prompt = f"""Act as an expert data ingestion pipeline assistant. Your sole task is to map column headers from a user's uploaded CSV file (`{file_name}`) to a predefined standard schema: `{target_schema}`. The raw headers are: `{raw_columns}`. Analyze the user's headers flexibly (case, symbols, abbreviations). Return ONLY a single raw JSON object mapping EACH standard header to a raw header, or to `null` if no match is found. Do not include any text or markdown formatting outside the JSON object."""
    response_text = get_gemini_response(prompt)
    if not response_text: return None
    try: return json.loads(response_text.strip().replace("```json", "").replace("```", ""))
    except json.JSONDecodeError: st.error(f"AI returned invalid mapping for {file_name}."); return None

def render_manual_column_mapper(file_name: str, raw_columns: list, target_schema: list) -> dict:
    """Creates a UI for manually mapping columns."""
    st.write(f"**Map columns for `{file_name}`:**")
    mapping, options = {}, [None] + raw_columns
    for std_col in target_schema:
        likely_match = next((col for col in raw_columns if std_col.lower().replace("_", "") in col.lower().replace("_", "")), None)
        idx = options.index(likely_match) if likely_match in options else 0
        selected = st.selectbox(f"Map for `{std_col}`:", options, index=idx, key=f"map_{file_name}_{std_col}")
        if selected: mapping[std_col] = selected
    return mapping

@st.cache_data
def load_and_normalize_data(uploaded_file_content: bytes, mapping: dict) -> pd.DataFrame:
    """Loads CSV data, renames columns, and performs data normalization."""
    df = pd.read_csv(io.BytesIO(uploaded_file_content))
    rename_map = {v: k for k, v in mapping.items() if v is not None and v in df.columns}
    df.rename(columns=rename_map, inplace=True)
    str_cols = ['station_name', 'state_name', 'district_name', 'agency_name', 'basin']
    for col in str_cols:
        if col in df.columns: df[col] = df[col].astype(str).str.strip().str.title()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
    numeric_cols = ['latitude', 'longitude', 'groundwaterlevel_mbgl', 'rainfall_mm', 'temperature_c', 'ph', 'turbidity_ntu', 'tds_ppm']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col == 'ph': df.loc[(df[col] < 0) | (df[col] > 14), col] = np.nan
            elif col in ['rainfall_mm', 'turbidity_ntu', 'tds_ppm']: df.loc[df[col] < 0, col] = np.nan
    return df

@st.cache_data
def generate_data_quality_report(df: pd.DataFrame, df_name: str) -> str:
    """Generates a detailed markdown report on data quality."""
    report = f"#### Quality Analysis: `{df_name}`\n- **Dimensions**: {df.shape[0]} rows & {df.shape[1]} columns.\n"
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100 if len(df) > 0 else 0
    missing_report = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if not missing_report.empty:
        report += "- **Missing Values Analysis**:\n"
        for col, pct in missing_report.items(): report += f"  - `{col}`: **{pct:.1f}% missing** ({missing[col]} values).\n"
    else: report += "- **Missing Values**: None detected. ✅\n"
    if 'timestamp' in df.columns and 'station_name' in df.columns:
        dupes = df.duplicated(subset=['timestamp', 'station_name']).sum()
        if dupes > 0: report += f"- **Duplicate Records**: Found **{dupes}** duplicate pairs. ⚠️\n"
    for col in ['groundwaterlevel_mbgl', 'rainfall_mm', 'temperature_c']:
        if col in df.columns and not df[col].dropna().empty:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
                if outliers > 0: report += f"- **Potential Outliers**: Detected {outliers} in `{col}`.\n"
    return report

def classify_files_with_heuristics(uploaded_files: list) -> tuple:
    """Intelligently classifies files based on column names."""
    file_contents = {f.name: f.getvalue() for f in uploaded_files}
    roles, candidates = {'timeseries': None, 'stations_gw': None, 'stations_rf': None}, []
    for file in uploaded_files:
        try:
            cols = pd.read_csv(io.BytesIO(file_contents[file.name]), nrows=0).columns.str.lower().str.replace('[_-]', '', regex=True)
            score = 0
            if any(ts in cols for ts in ['timestamp', 'date', 'datetime']): score += 10
            if 'latitude' in cols and 'longitude' in cols: score += 5
            if any(gw in cols for gw in ['groundwater', 'gwl', 'mbgl']): score += 2
            if any(rf in cols for rf in ['rainfall', 'rain']): score += 1
            candidates.append((score, file.name))
        except Exception: candidates.append((-1, file.name))
    candidates.sort(key=lambda x: x[0], reverse=True)
    if len(candidates) >= 1 and candidates[0][0] >= 10: roles['timeseries'] = candidates.pop(0)[1]
    if len(candidates) >= 1 and candidates[0][0] >= 5: roles['stations_gw'] = candidates.pop(0)[1]
    if len(candidates) >= 1 and candidates[0][0] >= 5: roles['stations_rf'] = candidates.pop(0)[1]
    return roles, file_contents

@st.cache_data
def calculate_long_term_station_trends(_ts_data: pd.DataFrame) -> pd.DataFrame:
    """Calculates long-term trends for all stations using Linear Regression."""
    trends = {}
    for name, group in _ts_data.groupby('station_name'):
        df_s = group.dropna(subset=['groundwaterlevel_mbgl', 'timestamp']).sort_values('timestamp')
        if len(df_s) > 30:
            df_s['time_ord'] = (df_s['timestamp'] - df_s['timestamp'].min()).dt.days
            model = LinearRegression().fit(df_s[['time_ord']], df_s['groundwaterlevel_mbgl'])
            trends[name] = model.coef_[0] * 365
    return pd.DataFrame(trends.items(), columns=['station_name', 'annual_trend_m'])

@st.cache_data
def analyze_monsoon_performance(_df_station: pd.DataFrame) -> pd.DataFrame:
    """Analyzes pre and post monsoon water levels for a single station."""
    df = _df_station.set_index('timestamp')
    df['year'] = df.index.year
    pre_monsoon = df[df.index.month.isin([3, 4, 5])].groupby('year')['groundwaterlevel_mbgl'].mean()
    post_monsoon = df[df.index.month.isin([10, 11, 12])].groupby('year')['groundwaterlevel_mbgl'].mean()
    monsoon_df = pd.DataFrame({'pre_monsoon_level_mbgl': pre_monsoon, 'post_monsoon_level_mbgl': post_monsoon}).dropna()
    monsoon_df['recharge_effect_m'] = monsoon_df['pre_monsoon_level_mbgl'] - monsoon_df['post_monsoon_level_mbgl']
    return monsoon_df.reset_index()

@st.cache_data
def detect_drought_events(_df_station: pd.DataFrame, percentile_threshold: int = 80) -> list:
    """Identifies historical drought periods."""
    if 'groundwaterlevel_mbgl' not in _df_station or _df_station['groundwaterlevel_mbgl'].dropna().empty: return []
    threshold = np.percentile(_df_station['groundwaterlevel_mbgl'].dropna(), percentile_threshold)
    df = _df_station[['timestamp', 'groundwaterlevel_mbgl']].copy()
    df['in_drought'] = df['groundwaterlevel_mbgl'] > threshold
    df['drought_block'] = (df['in_drought'].diff(1) != 0).astype('int').cumsum()
    periods = []
    for block in df[df['in_drought']]['drought_block'].unique():
        days = df[df['drought_block'] == block]
        duration = (days['timestamp'].max() - days['timestamp'].min()).days + 1
        if duration > 30: periods.append({'Start': days['timestamp'].min().date(), 'End': days['timestamp'].max().date(), 'Duration (Days)': duration, f'Peak Level (mbgl)': days['groundwaterlevel_mbgl'].max()})
    return periods

def haversine(lat1, lon1, lat2, lon2):
    R = 6371; lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def find_nearest_station(source_station, target_stations_df):
    """Finds the closest station from a DataFrame to a given source station."""
    if target_stations_df.empty or any(k not in source_station for k in ['latitude', 'longitude']) or 'latitude' not in target_stations_df.columns: return None, np.inf
    distances = haversine(source_station['latitude'], source_station['longitude'], target_stations_df['latitude'], target_stations_df['longitude'])
    return target_stations_df.loc[distances.idxmin()], distances.min()

@st.cache_data
def get_regional_status(ts, gw, col, quant):
    """Calculates regional groundwater status."""
    if gw.empty or col not in gw.columns: return pd.DataFrame()
    merged = pd.merge(ts, gw[['station_name', col]], on='station_name', how='left').dropna(subset=[col])
    if merged.empty: return pd.DataFrame()
    latest = merged.loc[merged.groupby('station_name')['timestamp'].idxmax()]
    thresholds = merged.groupby('station_name')['groundwaterlevel_mbgl'].quantile(quant)
    latest['threshold'] = latest['station_name'].map(thresholds)
    latest['status'] = np.where(latest['groundwaterlevel_mbgl'] > latest['threshold'], 'Low/Critical', 'Normal')
    return latest.groupby([col, 'status']).size().reset_index(name='count')

# --- 3. SIDEBAR / CONTROL PANEL UI ---

st.sidebar.title("⚙️ Control Panel")
st.sidebar.header("1. System Configuration")
st.sidebar.toggle("Disable All AI Features", key="ai_disabled", help="Run in manual mode without Gemini AI.")

# Use text_input to get the key and store it in session_state
if not st.session_state.ai_disabled:
    st.sidebar.text_input("Enter Gemini API Key:", type="password", key="gemini_key")

# Configure the API once at the start if the key exists in the session state
try:
    if st.session_state.get("gemini_key"):
        genai.configure(api_key=st.session_state.gemini_key)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")

st.sidebar.header("2. Data Ingestion")
uploaded_files = st.sidebar.file_uploader("Upload 3 CSVs (GW Stations, RF Stations, Time-Series)", type=['csv'], accept_multiple_files=True)

# Lottie animation for loading
if not uploaded_files:
    if LOTTIE_ANIMATION_DATA:
        st_lottie(LOTTIE_ANIMATION_DATA, height=300, key="loading_animation")
    st.info("👋 **Welcome to the National Water Intelligence Dashboard!** Please upload the three required CSV files to begin.")
    st.stop()
elif len(uploaded_files) != 3:
    st.info("👋 **Welcome!** Please ensure you upload exactly three CSV files to proceed.")
    st.stop()

# --- 4. CORE DATA PIPELINE LOGIC ---
probable_roles, file_contents = classify_files_with_heuristics(uploaded_files)
st.sidebar.info("Please confirm file assignments:")
all_fnames = [f.name for f in uploaded_files]
gw_station_fname = st.sidebar.selectbox("GW Station File:", all_fnames, index=all_fnames.index(probable_roles['stations_gw']) if probable_roles['stations_gw'] in all_fnames else 0)
rf_station_fname = st.sidebar.selectbox("RF Station File:", all_fnames, index=all_fnames.index(probable_roles['stations_rf']) if probable_roles['stations_rf'] in all_fnames else 1)
ts_fname = st.sidebar.selectbox("Time-Series Data File:", all_fnames, index=all_fnames.index(probable_roles['timeseries']) if probable_roles['timeseries'] in all_fnames else 2)

if len({gw_station_fname, rf_station_fname, ts_fname}) != 3: st.sidebar.error("Each file assignment must be unique."); st.stop()

# Initialize mapping variables
gw_mapping, rf_mapping, ts_mapping = None, None, None

with st.sidebar.expander("Column Mapping", expanded=False):
    schemas = {'station': ['station_name', 'latitude', 'longitude', 'state_name', 'district_name', 'agency_name', 'basin'], 'timeseries': ['station_name', 'timestamp', 'groundwaterlevel_mbgl', 'rainfall_mm', 'temperature_c', 'ph', 'turbidity_ntu', 'tds_ppm']}
    raw_cols = {name: pd.read_csv(io.BytesIO(file_contents[name]), nrows=0).columns.tolist() for name in file_contents}
    
    if not st.session_state.ai_disabled:
        st.subheader("🤖 AI-Assisted Mapping")
        if st.session_state.get("gemini_key"):
            gw_mapping = perform_ai_column_mapping(raw_cols[gw_station_fname], schemas['station'], gw_station_fname)
            rf_mapping = perform_ai_column_mapping(raw_cols[rf_station_fname], schemas['station'], rf_station_fname)
            ts_mapping = perform_ai_column_mapping(raw_cols[ts_fname], schemas['timeseries'], ts_fname)

            if gw_mapping: st.json({"GW Stations": gw_mapping}, expanded=False)
            if rf_mapping: st.json({"RF Stations": rf_mapping}, expanded=False)
            if ts_mapping: st.json({"Time-Series": ts_mapping}, expanded=False)
        else:
            st.warning("Please enter your Gemini API key to use AI-assisted mapping.")
    else:
        st.subheader("Manual Mapping")
        gw_mapping, rf_mapping, ts_mapping = render_manual_column_mapper(gw_station_fname, raw_cols[gw_station_fname], schemas['station']), render_manual_column_mapper(rf_station_fname, raw_cols[rf_station_fname], schemas['station']), render_manual_column_mapper(ts_fname, raw_cols[ts_fname], schemas['timeseries'])

# Final check for mapping completion
mapping_is_done = all([gw_mapping, rf_mapping, ts_mapping])
if not mapping_is_done:
    if not st.session_state.ai_disabled and not st.session_state.get("gemini_key"):
        st.info("👋 Please provide your Gemini API key in the sidebar to proceed with AI mapping, or disable AI for manual mapping.")
        st.stop()
    else:
        st.error("Column mapping is incomplete. Please complete the mapping in the sidebar.")
        st.stop()


gw_stations, rf_stations, ts_data = load_and_normalize_data(file_contents[gw_station_fname], gw_mapping), load_and_normalize_data(file_contents[rf_station_fname], rf_mapping), load_and_normalize_data(file_contents[ts_fname], ts_mapping)
ts_station_names = set(ts_data['station_name'].dropna().unique())
gw_stations_filtered, rf_stations_filtered = gw_stations[gw_stations['station_name'].isin(ts_station_names)].copy(), rf_stations[rf_stations['station_name'].isin(ts_station_names)].copy()

st.sidebar.success(f"Loaded data for {len(gw_stations_filtered)} GW & {len(rf_stations_filtered)} RF stations.")
with st.sidebar.expander("Data Quality Audit", expanded=True):
    st.markdown(generate_data_quality_report(gw_stations, gw_station_fname))
    st.markdown(generate_data_quality_report(rf_stations, rf_station_fname))
    st.markdown(generate_data_quality_report(ts_data, ts_fname))

# --- 5. HIERARCHICAL SIDEBAR FILTERS ---
def reset_downstream_cache(): st.session_state.pop('forecast_results', None); st.session_state.pop('report_data', None)
st.sidebar.header("3. Analysis Filters")
ALL = "All"
all_stations_ui = pd.concat([gw_stations_filtered[['state_name', 'district_name', 'station_name', 'basin']], rf_stations_filtered[['state_name', 'district_name', 'station_name', 'basin']]]).drop_duplicates().sort_values('state_name')
state = st.sidebar.selectbox("State:", [ALL] + sorted(all_stations_ui['state_name'].unique()), on_change=reset_downstream_cache, key='state_filter')
if state != ALL: all_stations_ui = all_stations_ui[all_stations_ui['state_name'] == state]
district = st.sidebar.selectbox("District:", [ALL] + sorted(all_stations_ui['district_name'].unique()), on_change=reset_downstream_cache, key='district_filter')
if district != ALL: all_stations_ui = all_stations_ui[all_stations_ui['district_name'] == district]
basin = st.sidebar.selectbox("River Basin:", [ALL] + sorted(all_stations_ui['basin'].dropna().unique()), on_change=reset_downstream_cache, key='basin_filter')
if basin != ALL: all_stations_ui = all_stations_ui[all_stations_ui['basin'] == basin]
station_name = st.sidebar.selectbox("GW Station:", [ALL] + sorted(all_stations_ui['station_name'].unique()), on_change=reset_downstream_cache, key='station_filter')

st.sidebar.header("4. Time Range Filter")
time_range_options = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 6 Months": 180, "Last Year": 365, "All Time": None}
selected_range_label = st.sidebar.selectbox("Time Range:", list(time_range_options.keys()), on_change=reset_downstream_cache, key='time_filter')
days_to_filter = time_range_options[selected_range_label]

# --- 6. DYNAMIC DATA PROCESSING ---
stations_in_scope = gw_stations_filtered.copy()
if state != ALL: stations_in_scope = stations_in_scope[stations_in_scope['state_name'] == state]
if district != ALL: stations_in_scope = stations_in_scope[stations_in_scope['district_name'] == district]
if basin != ALL: stations_in_scope = stations_in_scope[stations_in_scope['basin'] == basin]
single_station_mode = (station_name != ALL)
if single_station_mode: stations_in_scope = stations_in_scope[stations_in_scope['station_name'] == station_name]
if stations_in_scope.empty: st.warning("No GW stations match current filters."); st.stop()

df_base_filtered = ts_data[ts_data['station_name'].isin(stations_in_scope['station_name'])].copy()
if days_to_filter: df_filtered = df_base_filtered[df_base_filtered['timestamp'] >= (df_base_filtered['timestamp'].max() - pd.Timedelta(days=days_to_filter))]
else: df_filtered = df_base_filtered
if df_filtered.empty: st.warning("No time-series data in selected time range."); st.stop()

# --- 7. MAIN APPLICATION UI & TABS ---
st.markdown('<div class="dashboard-title"><p class="gradient-text">National Water Resources Intelligence Dashboard</p></div>', unsafe_allow_html=True)
tabs = ["🗺️ Unified Map", "📊 At-a-Glance", "⚖️ Policy", "🏛️ Strategic Planning", "🔬 Research Hub", "💧 Public Info", "🌊 Advanced Hydrology", "📋 Full Report"]
if 'active_tab' not in st.session_state: st.session_state.active_tab = tabs[0]
def set_active_tab(): st.session_state.active_tab = st.session_state.navigation_radio
try: default_tab_index = tabs.index(st.session_state.active_tab)
except ValueError: default_tab_index = 0
st.radio("Main Navigation", tabs, index=default_tab_index, key="navigation_radio", on_change=set_active_tab, horizontal=True) #, label_visibility="collapsed")
selected_tab = st.session_state.active_tab

main_container = st.container()
with main_container:
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

    # --- TAB 1: Unified Map View ---
    if selected_tab == tabs[0]:
        st.header("🗺️ Unified Geographic Network View")
        st.write("Visualize the geographic distribution and density of monitoring stations.")
        c1, c2 = st.columns([1, 2])
        map_view_type = c1.radio("Map Style:", ["Points of Interest", "Heatmap (Density)"])
        available_years = sorted(ts_data['timestamp'].dt.year.unique(), reverse=True)
        selected_year = c2.selectbox("Yearly Status:", ["All Time"] + available_years)
        map_df_gw = stations_in_scope.copy()
        map_df_rf = rf_stations_filtered[rf_stations_filtered['state_name'].isin(map_df_gw['state_name'].unique())].copy()
        info_text = "🔵 GW Stations | 🟢 RF Stations"
        
        if selected_year != "All Time":
            ts_year_data = ts_data[ts_data['timestamp'].dt.year == selected_year]
            status_for_year = get_regional_status(ts_year_data, gw_stations_filtered, 'station_name', 0.75)
            status_map = status_for_year.set_index('station_name')['status'].to_dict()
            map_df_gw['status'] = map_df_gw['station_name'].map(status_map).fillna('No Data')
            color_map = {'Low/Critical': '#FF0000', 'Normal': '#0000FF', 'No Data': '#808080'}
            map_df_gw['color'] = map_df_gw['status'].map(color_map)
            info_text = f"Status in {selected_year}: 🔵 Normal | 🔴 Low/Critical | ⚫ No Data"
        else:
            map_df_gw['color'] = '#0066FF'
        
        map_df_rf['color'] = '#00CC66'
        map_df_gw['size'], map_df_rf['size'] = 25, 25
        
        if single_station_mode and not stations_in_scope.empty:
            selected_gw_station = stations_in_scope.iloc[0]
            map_df_gw.loc[map_df_gw['station_name'] == station_name, 'color'], map_df_gw.loc[map_df_gw['station_name'] == station_name, 'size'] = '#FFD700', 100
            if not map_df_rf.empty:
                nearest_rf_station, distance = find_nearest_station(selected_gw_station, map_df_rf)
                if nearest_rf_station is not None:
                    map_df_rf.loc[map_df_rf['station_name'] == nearest_rf_station['station_name'], 'color'], map_df_rf.loc[map_df_rf['station_name'] == nearest_rf_station['station_name'], 'size'] = '#FFA500', 100
                info_text += f" | ⭐ Selected GW | 🟠 Nearest RF ({distance:.2f} km)"
        
        map_df = pd.concat([map_df_gw, map_df_rf]).reset_index(drop=True).dropna(subset=['latitude', 'longitude'])
        
        if map_view_type == "Heatmap (Density)":
             st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/dark-v9', initial_view_state=pdk.ViewState(latitude=map_df['latitude'].mean(), longitude=map_df['longitude'].mean(), zoom=5, pitch=50), layers=[pdk.Layer('HexagonLayer', data=map_df, get_position='[longitude, latitude]', radius=8000, elevation_scale=100, extruded=True, pickable=True)]))
        else:
            st.map(map_df, latitude='latitude', longitude='longitude', color='color', size='size')
        st.info(info_text)

    # --- TAB 2: At-a-Glance Dashboard ---
    elif selected_tab == tabs[1]:
        st.header("📊 At-a-Glance Dashboard")
        st.write("Get a high-level overview of key water metrics for the selected region and time period.")
        agency = st.selectbox("Filter by Agency:", [ALL] + sorted(stations_in_scope['agency_name'].unique().tolist()))
        df_tab = df_filtered.copy()
        if agency != ALL:
            stations_tab = stations_in_scope[stations_in_scope['agency_name'] == agency]
            df_tab = df_filtered[df_filtered['station_name'].isin(stations_tab['station_name'])]
        if df_tab.empty:
            st.warning(f"No data for agency: {agency}.")
            st.stop()
        
        with card("Key Performance Indicators"):
            cols = st.columns(4)
            if 'groundwaterlevel_mbgl' in df_tab.columns and not df_tab['groundwaterlevel_mbgl'].dropna().empty:
                gwl = df_tab['groundwaterlevel_mbgl'].dropna()
                animated_metric(cols[0], "Avg GW Level", gwl.mean(), "m")
                delta_val = gwl.iloc[-1] - gwl.iloc[0] if len(gwl) > 1 else None
                animated_metric(cols[1], "Most Recent GW Level", gwl.iloc[-1], "m", delta=f"{delta_val:.2f} m" if delta_val is not None else None, delta_color="inverse")
            if 'rainfall_mm' in df_tab.columns and not df_tab['rainfall_mm'].dropna().empty: animated_metric(cols[2], "Total Rainfall", df_tab['rainfall_mm'].sum(), "mm")
            if 'temperature_c' in df_tab.columns and not df_tab['temperature_c'].dropna().empty: animated_metric(cols[3], "Avg Temperature", df_tab['temperature_c'].mean(), "°C")
            
            cols2 = st.columns(4) # Place other metrics in a new row if needed
            if 'ph' in df_tab.columns and not df_tab['ph'].dropna().empty: animated_metric(cols2[0], "Avg pH", df_tab['ph'].mean())
            if 'turbidity_ntu' in df_tab.columns and not df_tab['turbidity_ntu'].dropna().empty: animated_metric(cols2[1], "Latest Turbidity", df_tab['turbidity_ntu'].dropna().iloc[-1], "NTU")
            if 'tds_ppm' in df_tab.columns and not df_tab['tds_ppm'].dropna().empty: animated_metric(cols2[2], "Latest TDS", df_tab['tds_ppm'].dropna().iloc[-1], "ppm")

        with card("Agency Contribution"):
            agency_dist = stations_in_scope['agency_name'].value_counts()
            pull = [0.2 if idx == agency else 0 for idx in agency_dist.index]
            fig_pie = px.pie(agency_dist, values=agency_dist.values, names=agency_dist.index, title='Monitored Stations by Agency', hole=0.4)
            fig_pie.update_traces(pull=pull, textinfo='percent+label').update_layout(transition_duration=500, template="streamlit")
            st.plotly_chart(fig_pie, use_container_width=True)


    # --- TAB 3: Policy & Governance ---
    elif selected_tab == tabs[2]:
        st.header("⚖️ Policy & Governance Insights")
        st.write("Analyze regional water stress and long-term trends to inform policy decisions.")
        
        with card("Regional Groundwater Stress Hotspots"):
            level = st.radio("Analyze by:", ("State", "River Basin"), horizontal=True)
            group_col = 'state_name' if level == "State" else 'basin'
            percentile = st.slider("Define 'Critical' Level (%):", 50, 95, 75, 5)
            status = get_regional_status(ts_data, gw_stations_filtered, group_col, percentile / 100.0)
            if not status.empty:
                chart = alt.Chart(status).mark_bar().encode(
                    x=alt.X(f'{group_col}:N', sort='-y', title=level, axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('count:Q', title="Number of Stations"),
                    color=alt.Color('status:N', scale=alt.Scale(domain=['Normal', 'Low/Critical'], range=['#5cb85c', '#d9534f'])),
                    tooltip=[group_col, 'status', 'count']
                ).properties(
                    title=f'Groundwater Status by {level}'
                ).configure_title(fontSize=36).interactive()
                st.altair_chart(chart, use_container_width=True)
                if st.button("Generate AI Policy Briefing", disabled=not st.session_state.get("gemini_key")):
                    analysis = get_gemini_response(f"""As a senior water policy advisor, analyze this data on GW stress by {level}: {status.to_json(orient='records')}. Provide a briefing with: executive summary, key hotspots, 3 actionable policy recommendations, and data gaps.""")
                    if analysis:
                        st.markdown(analysis)

        with card("Long-Term Station Health Trends"):
            with st.spinner("Analyzing long-term trends..."):
                trends = calculate_long_term_station_trends(ts_data)
            if not trends.empty:
                c1, c2 = st.columns(2)
                c1.error("Top 5 Declining Stations")
                c1.dataframe(trends.sort_values('annual_trend_m', ascending=False).head(5))
                c2.success("Top 5 Improving Stations")
                c2.dataframe(trends.sort_values('annual_trend_m').head(5))
            else:
                st.info("No stations with sufficient historical data (>30 points) for long-term trend analysis were found in the current dataset.")

        with card("AI Policy Simulator & Advisor"):
            if not st.session_state.ai_disabled:
                policy_goal = st.selectbox("Select Policy Goal:", ["Increase Groundwater Recharge", "Reduce Water Depletion", "Improve Drought Resilience"])
                if st.button(f"Generate Policy Brief for '{policy_goal}'", disabled=not st.session_state.get("gemini_key")):
                    analysis = get_gemini_response(f"""As a water policy expert for India, create a detailed policy brief on **"{policy_goal}"**. Structure it with: Introduction, Key Challenges, 3-5 Strategic Interventions, Implementation Roadmap, and Conclusion.""")
                    if analysis:
                        st.markdown(analysis)
            else:
                st.info("Enable AI to use the Policy Simulator.")

    # --- TAB 4: Strategic Planning ---
    elif selected_tab == tabs[3]:
        st.header("🏛️ Strategic Planning & Scenario Modeling")
        st.write("Conduct long-term supply vs. demand analysis for a specific station's area of influence.")
        if not single_station_mode:
            st.info("Select a single station for planning tools.")
            st.stop()
        station_ts_data = df_base_filtered[df_base_filtered['station_name'] == station_name]
        
        with card(f"Long-Term Historical Trend for {station_name}"):
            if not station_ts_data.empty:
                line = alt.Chart(station_ts_data).mark_line(point=True).encode(x='timestamp:T', y=alt.Y('groundwaterlevel_mbgl:Q', title="GW Level (mbgl)"), tooltip=['timestamp', 'groundwaterlevel_mbgl']).properties(title='Historical Groundwater Level').interactive()
                st.altair_chart(line, use_container_width=True)

        with card("Sustainable Yield & Demand Modeling"):
            st.subheader("2. Sustainable Yield Estimation")
            sy_planning = st.number_input("Specific Yield (Sy):", 0.01, 0.50, 0.15, 0.01)
            planning_df = station_ts_data.set_index('timestamp').asfreq('D').interpolate(method='time')
            planning_df['gw_level_change'] = planning_df['groundwaterlevel_mbgl'].diff()
            planning_df['recharge_mm'] = planning_df.apply(lambda r: (r['gw_level_change'] * -1 * sy_planning * 1000) if r['gw_level_change'] < 0 else 0, axis=1)
            avg_annual_recharge = planning_df['recharge_mm'].resample('Y').sum().mean()
            sustainable_yield_mm = avg_annual_recharge * 0.7
            c1, c2 = st.columns(2)
            c1.metric("Avg Est. Annual Recharge", f"{avg_annual_recharge:.2f} mm/year")
            c2.metric("Est. Sustainable Yield", f"{sustainable_yield_mm:.2f} mm/year")
            st.subheader(f"3. Water Balance Scenario Modeling")
            area = st.number_input("Area of Influence (sq. km):", 0.1, 1000.0, 10.0, 1.0)
            sustainable_volume_m3 = (sustainable_yield_mm / 1000) * (area * 1_000_000)
            st.subheader("4. Demand Modeling")
            c5, c6, c7 = st.columns(3)
            agri_demand_m3 = c5.number_input("Agri. Demand (m³):", value=50000)
            ind_demand_m3 = c6.number_input("Ind. Demand (m³):", value=20000)
            dom_demand_m3 = c7.number_input("Dom. Demand (m³):", value=30000)
            total_demand = agri_demand_m3 + ind_demand_m3 + dom_demand_m3
            balance = sustainable_volume_m3 - total_demand
            st.subheader("5. Water Balance Results")
            fig_balance = go.Figure(data=[go.Bar(name='Sustainable Supply', x=['Water Balance'], y=[sustainable_volume_m3]), go.Bar(name='Projected Demand', x=['Water Balance'], y=[total_demand])])
            fig_balance.update_layout(barmode='group', title='Annual Supply vs. Demand', template='plotly_dark', transition_duration=500)
            st.plotly_chart(fig_balance, use_container_width=True)
            if balance > 0:
                st.success(f"**Projected Surplus: {balance:,.0f} m³/year**")
            else:
                st.error(f"**Projected Deficit: {abs(balance):,.0f} m³/year**")
            if st.button("Generate AI Strategic Recommendations", disabled=not st.session_state.get("gemini_key")):
                analysis = get_gemini_response(f"""As a water consultant, analyze this scenario for {station_name}: Supply={sustainable_volume_m3:,.0f} m³, Demand={total_demand:,.0f} m³, Balance={balance:,.0f} m³/year. Provide recommendations for a deficit or surplus.""")
                if analysis:
                    st.markdown(analysis)

    # --- TAB 5: Research Hub ---
    elif selected_tab == tabs[4]:
        st.header("🔬 Research Hub & Advanced Analytics")
        st.write("Dive deep into water quality analysis, parameter correlations, and predictive forecasting.")

        with card("1. Comprehensive Water Quality Analysis"):
            df_plot = df_filtered if single_station_mode else df_filtered.groupby('timestamp').mean(numeric_only=True).reset_index()
            fig_quality = make_subplots(specs=[[{"secondary_y": True}]])
            if 'ph' in df_plot.columns: fig_quality.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['ph'], name='pH'), secondary_y=False)
            if 'tds_ppm' in df_plot.columns: fig_quality.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['tds_ppm'], name='TDS (ppm)'), secondary_y=True)
            if 'turbidity_ntu' in df_plot.columns: fig_quality.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['turbidity_ntu'], name='Turbidity (NTU)'), secondary_y=True)
            fig_quality.update_layout(title_text="Water Quality Parameters Over Time", template='plotly_dark', transition_duration=500)
            fig_quality.update_yaxes(title_text="pH Level", secondary_y=False); fig_quality.update_yaxes(title_text="TDS / Turbidity", secondary_y=True)
            st.plotly_chart(fig_quality, use_container_width=True)

        with card("2. AI Correlation Analyst"):
            if not st.session_state.ai_disabled:
                cols = [col for col in ['groundwaterlevel_mbgl', 'rainfall_mm', 'temperature_c', 'ph', 'turbidity_ntu', 'tds_ppm'] if col in df_filtered.columns]
                c1, c2 = st.columns(2); p1 = c1.selectbox("Parameter 1:", cols); p2 = c2.selectbox("Parameter 2:", cols, index=1 if len(cols)>1 else 0)
                if st.button("Analyze Correlation with AI", disabled=not st.session_state.get("gemini_key")):
                    corr_df = df_filtered[[p1, p2]].dropna()
                    if len(corr_df) > 1:
                        corr = corr_df.corr().iloc[0, 1]
                        st.metric(f"Pearson Correlation between {p1} and {p2}", f"{corr:.3f}")
                        prompt = f"""As a research hydrologist, analyze the relationship between '{p1}' and '{p2}' (Pearson correlation: {corr:.2f}). Explain the correlation, its hydrological context, and research implications."""
                        analysis = get_gemini_response(prompt)
                        if analysis:
                            st.markdown(analysis)
                    else: st.warning("Not enough overlapping data for correlation.")
            else: st.info("Enable AI to use the Correlation Analyst.")

        with card("3. High-Accuracy Predictive Forecast"):
            st.warning("Note: This forecast uses a generalized SARIMAX model for rapid analysis. For scientific or operational use, model parameters should be tuned specifically for each time-series dataset.")
            if not single_station_mode:
                st.info("Select a single station for forecasting.")
            else:
                days = st.slider("Days to forecast:", 7, 180, 30)
                if st.button("Generate Forecast", disabled=not st.session_state.get("gemini_key")):
                    with st.spinner("Running SARIMAX model..."):
                        try:
                            df_f = df_filtered[['timestamp', 'groundwaterlevel_mbgl']].set_index('timestamp').asfreq('D').interpolate(method='time')
                            if len(df_f) < 24: st.error("Forecasting failed: Requires at least 24 data points.")
                            else:
                                fit = SARIMAX(df_f['groundwaterlevel_mbgl'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
                                pred, pred_ci = fit.get_forecast(steps=days), fit.get_forecast(steps=days).conf_int()
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=df_f.index, y=df_f['groundwaterlevel_mbgl'], name='Historical Data', line=dict(color='cyan')))
                                fig.add_trace(go.Scatter(x=pred.predicted_mean.index, y=pred.predicted_mean, name='Forecast', line=dict(color='orange', dash='dot')))
                                fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:, 1], fill=None, mode='lines', line_color='rgba(255,165,0,0.3)', showlegend=False))
                                fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:, 0], fill='tonexty', name='95% Confidence Interval', mode='lines', line_color='rgba(255,165,0,0.3)'))
                                fig.update_layout(title=f"Water Level Forecast for {station_name}", template='plotly_dark', transition_duration=500)
                                st.plotly_chart(fig, use_container_width=True)
                                st.session_state['forecast_results'] = pred.summary_frame(); st.dataframe(st.session_state['forecast_results'])
                        except Exception as e: st.error(f"Forecasting failed. Error: {e}")


    # --- TAB 6: Public Info ---
    elif selected_tab == tabs[5]:
        st.header(f"💧 Public Water Information Center")
        st.write("Understand the current water situation in your area with simple, easy-to-read gauges.")
        
        with card("Current Water Status Gauges"):
            c1, c2, c3 = st.columns(3)
            with c1:
                if 'groundwaterlevel_mbgl' in df_filtered.columns and not df_filtered.dropna(subset=['groundwaterlevel_mbgl']).empty:
                    latest, avg = df_filtered.dropna(subset=['groundwaterlevel_mbgl']).iloc[-1]['groundwaterlevel_mbgl'], df_filtered['groundwaterlevel_mbgl'].mean()
                    p10, p90 = df_filtered['groundwaterlevel_mbgl'].quantile(0.1), df_filtered['groundwaterlevel_mbgl'].quantile(0.9)
                    fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=latest, title={'text': "GW Level (mbgl)"}, gauge={'axis': {'range': [p10, p90]}, 'steps': [{'range': [p10, avg], 'color': "lightgreen"}, {'range': [avg, p90], 'color': "lightcoral"}]}))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                else: st.info("No GW Level data.")
            with c2:
                if 'ph' in df_filtered.columns and not df_filtered.dropna(subset=['ph']).empty:
                    latest_ph = df_filtered.dropna(subset=['ph']).iloc[-1]['ph']
                    fig_ph = go.Figure(go.Indicator(mode="gauge+number", value=latest_ph, title={'text': "Water pH Level"}, gauge={'axis': {'range': [5, 9]}, 'steps': [{'range': [5, 6.5], 'color': "lightcoral"}, {'range': [6.5, 8.5], 'color': 'lightgreen'}, {'range': [8.5, 9], 'color': 'lightcoral'}]}))
                    st.plotly_chart(fig_ph, use_container_width=True)
                else: st.info("No pH data.")
            with c3:
                if 'tds_ppm' in df_filtered.columns and not df_filtered.dropna(subset=['tds_ppm']).empty:
                    latest_tds = df_filtered.dropna(subset=['tds_ppm']).iloc[-1]['tds_ppm']
                    fig_tds = go.Figure(go.Indicator(mode="gauge+number", value=latest_tds, title={'text': "TDS (ppm)"}, gauge={'axis': {'range': [0, 1000]}, 'steps': [{'range': [0, 500], 'color': "lightgreen"}, {'range': [500, 1000], 'color': 'lightcoral'}]}))
                    st.plotly_chart(fig_tds, use_container_width=True)
                else: st.info("No TDS data.")
            st.info("Green zones on gauges indicate desirable ranges, while red indicates levels that may require attention.")

        with card("💡 What This Means For You & AI-Powered Advice"):
            if not st.session_state.ai_disabled:
                if st.button("Get a Simple AI Summary of the Water Situation", disabled=not st.session_state.get("gemini_key")):
                    prompt_data = {'GW Level': f"{df_filtered['groundwaterlevel_mbgl'].dropna().iloc[-1]:.2f} mbgl" if 'groundwaterlevel_mbgl' in df_filtered.columns and not df_filtered['groundwaterlevel_mbgl'].dropna().empty else "N/A", 'pH': f"{df_filtered['ph'].dropna().iloc[-1]:.2f}" if 'ph' in df_filtered.columns and not df_filtered['ph'].dropna().empty else "N/A", 'TDS': f"{df_filtered['tds_ppm'].dropna().iloc[-1]:.2f} ppm" if 'tds_ppm' in df_filtered.columns and not df_filtered['tds_ppm'].dropna().empty else "N/A"}
                    prompt = f"""As a public information assistant, explain the local water situation in simple, non-technical language based on: {prompt_data}. Explain water availability (higher mbgl is worse), quality (pH ideal 6.5-8.5, TDS ideal < 500 ppm), give a one-sentence summary, and a daily tip related to the data."""
                    analysis = get_gemini_response(prompt)
                    if analysis:
                        st.info(analysis)
            else: st.info("Enable AI features for a simple summary.")

    # --- TAB 7: Advanced Hydrology ---
    elif selected_tab == tabs[6]:
        st.header("🌊 Advanced Hydrology Analysis")
        st.write("Explore detailed hydrological characteristics like volatility, seasonal performance, and historical drought events.")
        if not single_station_mode: st.info("Select a single station for advanced hydrology tools."); st.stop()
        df_hydro = df_filtered.set_index('timestamp').asfreq('D').interpolate(method='time')

        with card("1. Water Level Fluctuation & Volatility (90-Day Rolling Window)"):
            df_hydro['90_day_avg'] = df_hydro['groundwaterlevel_mbgl'].rolling(90).mean(); df_hydro['volatility'] = df_hydro['groundwaterlevel_mbgl'].rolling(90).std()
            fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
            fig_vol.add_trace(go.Scatter(x=df_hydro.index, y=df_hydro['groundwaterlevel_mbgl'], name='Daily Level'), secondary_y=False)
            fig_vol.add_trace(go.Scatter(x=df_hydro.index, y=df_hydro['90_day_avg'], name='90-Day Avg Trend', line=dict(dash='dot', color='orange')), secondary_y=False)
            fig_vol.add_trace(go.Scatter(x=df_hydro.index, y=df_hydro['volatility'], name='90-Day Volatility', line=dict(color='lightgreen')), secondary_y=True)
            fig_vol.update_layout(title="Water Level vs. 90-Day Volatility", template='plotly_dark', transition_duration=500); fig_vol.update_yaxes(title_text="GW Level (mbgl)", secondary_y=False); fig_vol.update_yaxes(title_text="Volatility (Std. Dev.)", secondary_y=True)
            st.plotly_chart(fig_vol, use_container_width=True)

        with card("2. Smoothed Trend Analysis (EWMA)"):
            ewma_span = st.slider("Select EWMA Span (days):", 7, 180, 30, key='ewma_span')
            df_hydro['ewma'] = df_hydro['groundwaterlevel_mbgl'].ewm(span=ewma_span, adjust=False).mean()
            fig_ewma = px.line(df_hydro, y=['groundwaterlevel_mbgl', 'ewma'], title="Exponentially Weighted Moving Average Trend", template='plotly_dark')
            fig_ewma.update_layout(transition_duration=500)
            st.plotly_chart(fig_ewma, use_container_width=True)

        with card("3. Seasonal Aquifer Performance (Pre- vs. Post-Monsoon)"):
            monsoon = analyze_monsoon_performance(df_base_filtered)
            if not monsoon.empty:
                st.metric("Average Monsoon Recharge Effect", f"{monsoon['recharge_effect_m'].mean():.2f} m")
                fig_monsoon = px.bar(monsoon, x='year', y=['pre_monsoon_level_mbgl', 'post_monsoon_level_mbgl'], barmode='group', title='Pre vs. Post Monsoon Water Levels', template='plotly_dark')
                fig_monsoon.update_layout(transition_duration=500); st.plotly_chart(fig_monsoon, use_container_width=True); st.dataframe(monsoon)
            else: st.info("Insufficient seasonal data for monsoon analysis.")

        with card("4. Historical Drought Event Analysis"):
            drought_p = st.slider("Define Drought Threshold (% of deepest historical levels):", 70, 99, 85)
            droughts = detect_drought_events(df_base_filtered, drought_p)
            if droughts: st.warning(f"Detected {len(droughts)} significant drought periods."); st.dataframe(pd.DataFrame(droughts))
            else: st.success("No significant drought periods detected.")


    # --- TAB 8: Full Report ---
    elif selected_tab == tabs[7]:
        st.header("📋 Generate Consolidated Intelligence Report")
        st.write("Compile all key findings from the selected data into a single, downloadable report.")
        if st.button("➕ Generate Full Report for Current Selection"):
            with st.spinner("Compiling data and insights..."):
                report = {"report_generated_on": datetime.now().isoformat(), "selection_filters": {"State": state, "District": district, "Basin": basin, "Station": station_name, "Time Period": selected_range_label}, "key_performance_indicators": {}, "long_term_trends": "N/A for multiple stations.", "forecast_summary": "Not generated.", "policy_summary": {}}
                if 'groundwaterlevel_mbgl' in df_filtered: report["key_performance_indicators"]["avg_gw_level_mbgl"] = f"{df_filtered['groundwaterlevel_mbgl'].mean():.2f}"
                if 'rainfall_mm' in df_filtered: report["key_performance_indicators"]["total_rainfall_mm"] = f"{df_filtered['rainfall_mm'].sum():.2f}"
                if single_station_mode:
                    trends = calculate_long_term_station_trends(df_base_filtered)
                    if not trends.empty: report["long_term_trends"] = f"{trends.iloc[0]['annual_trend_m']:.3f} m/year"
                status_report = get_regional_status(ts_data, gw_stations_filtered, 'state_name', 0.75)
                report['policy_summary'] = status_report.to_dict('records')
                if 'forecast_results' in st.session_state:
                    forecast_df = st.session_state['forecast_results']
                    report["forecast_summary"] = {"days_forecasted": len(forecast_df), "final_predicted_level": f"{forecast_df['mean'].iloc[-1]:.2f} mbgl"}
                st.session_state['report_data'] = report; st.success("Report generated successfully!")
        if 'report_data' in st.session_state:
            report_data = st.session_state['report_data']
            st.subheader("Consolidated Report"); st.markdown("---")
            st.markdown("#### Selection Criteria"); st.json(report_data['selection_filters'])
            st.markdown("#### Summary Metrics"); st.json(report_data['key_performance_indicators'])
            st.markdown("#### Key Insights")
            c1, c2 = st.columns(2)
            top_stressed_df = pd.DataFrame(report_data.get('policy_summary', [])).sort_values('count', ascending=False)
            top_stressed = top_stressed_df.iloc[0]['state_name'] if not top_stressed_df.empty else "N/A"
            c1.metric("Top Stressed State (Example)", top_stressed)
            c2.metric("Long-Term Trend", report_data["long_term_trends"])
            st.metric("Short-Term Forecast", report_data["forecast_summary"] if isinstance(report_data["forecast_summary"], str) else report_data["forecast_summary"].get("final_predicted_level", "N/A"))
            st.markdown("---")
            st.download_button(label="📥 Download Full Report (JSON)", data=json.dumps(report_data, indent=2), file_name=f"water_report.json", mime="application/json")
            if st.button("Generate AI Executive Summary of Report", disabled=not st.session_state.get("gemini_key")):
                prompt = f"""Analyze the following JSON report on water resources and generate a concise executive summary for a high-level government official. Focus on the most critical findings and actionable insights. Report Data: {json.dumps(st.session_state['report_data'])}"""
                summary = get_gemini_response(prompt)
                if summary:
                    st.subheader("🤖 AI-Generated Executive Summary"); st.markdown(summary)

    st.markdown("</div>", unsafe_allow_html=True) # End fade-in wrapper

