# app.py
# National Water Resources Intelligence Dashboard
# A comprehensive Streamlit application for visualizing, analyzing, and forecasting water resource data.
# This application integrates Google's Gemini AI for advanced data mapping, analysis, and reporting.

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import warnings
import io
import json
import pydeck as pdk # Added for heatmap functionality
from sklearn.linear_model import LinearRegression # Added for trend analysis

# --- Page Configuration ---
st.set_page_config(layout='wide', page_title='National Water Resources Intelligence Dashboard')

# --- Suppress Warnings ---
warnings.filterwarnings("ignore")

# --- Gemini API Configuration ---
st.sidebar.title("Configuration")

# --- NEW: AI Feature Toggle ---
st.sidebar.toggle("Disable All AI Features", key="ai_disabled", help="Toggle this to run the app without AI. You will need to map data columns manually.")

gemini_api_key = None
if not st.session_state.get('ai_disabled', False):
    gemini_api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password", key="gemini_key")
    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
        except Exception as e:
            st.error(f"Error configuring Gemini API: {e}")
else:
    st.sidebar.info("AI features are off. Column mapping will be manual.")


# -------------------- Helper Functions --------------------

# --- Corrected Code ---

# --- Corrected Code ---
def get_gemini_analysis(prompt, api_key): # Add api_key as an argument
    """Generic function to call the Gemini API."""
    if st.session_state.get('ai_disabled', False):
        st.warning("AI features are currently disabled by the user.")
        return None
    
    # Check the passed-in api_key argument directly
    if not api_key:
        st.warning("Please enter your Gemini API key for AI analysis.")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        with st.spinner("🤖 Calling the AI... Please wait."):
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Could not get analysis from Gemini. Error: {e}")
        return None
# --- Corrected Code ---
@st.cache_data
def get_ai_column_mapping(raw_columns, target_schema, api_key): # Add api_key
    """
    Uses AI to map raw CSV columns to the application's standard schema.
    """
    prompt = f"""
    You are an expert data mapping AI. Your task is to map a list of raw column headers from a user's CSV file to a standard schema.
    The standard schema headers are: {target_schema}
    The user's raw CSV headers are: {raw_columns}

    Analyze the user's headers and return a JSON object that maps EACH standard schema header to the corresponding raw header from the user's list.
    - Be flexible with variations like case, spacing, symbols ('_','-'), and common abbreviations (e.g., 'lat' for 'latitude').
    - If a standard header has NO corresponding match in the user's list, map it to null.
    - Your entire response must be ONLY the JSON object, with no other text or formatting.
    """
    response_text = get_gemini_analysis(prompt, api_key) # Pass the api_key down
    if not response_text:
        return None
    try:
        json_str = response_text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_str)
    except json.JSONDecodeError:
        st.error("AI returned an invalid mapping format. Please check your file columns and try again.")
        return None

# --- NEW: Manual Column Mapping Function ---
def manual_column_mapper(file_name, raw_columns, target_schema):
    """Creates a UI for manually mapping columns."""
    st.write(f"**Manually map columns for `{file_name}`:**")
    mapping = {}
    options = [None] + raw_columns
    for standard_col in target_schema:
        # Try to find a likely match to set as default index
        try:
            cleaned_standard = standard_col.lower().replace("_", "")
            likely_matches = [i for i, col in enumerate(raw_columns) if cleaned_standard in col.lower().replace("_", "")]
            default_index = likely_matches[0] + 1 if likely_matches else 0 # +1 for the None option
        except Exception:
            default_index = 0

        selected_col = st.selectbox(
            f"Which column is `{standard_col}`?",
            options=options,
            index=default_index,
            key=f"map_{file_name}_{standard_col}"
        )
        if selected_col:
            mapping[standard_col] = selected_col
    return mapping


# --- Main Data Loading and Processing Functions ---

@st.cache_data
def load_and_normalize_data(uploaded_file_content, ai_mapping):
    """
    Loads CSV content, renames columns based on AI mapping, and normalizes data.
    """
    df = pd.read_csv(io.BytesIO(uploaded_file_content))
    
    # Build the rename map robustly, handling strings or lists from the AI
    rename_map = {}
    if ai_mapping:
        for standard_col, raw_col_val in ai_mapping.items():
            if isinstance(raw_col_val, str) and raw_col_val in df.columns:
                rename_map[raw_col_val] = standard_col
            elif isinstance(raw_col_val, list):
                for col_option in raw_col_val:
                    if col_option in df.columns:
                        rename_map[col_option] = standard_col
                        break
    
    df.rename(columns=rename_map, inplace=True)
    
    str_cols = ['station_name', 'state_name', 'district_name', 'agency_name', 'basin']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    numeric_cols = ['groundwaterlevel_mbgl', 'rainfall_mm', 'temperature_c', 'ph', 'turbidity_ntu', 'tds_ppm']
    for col in numeric_cols:
        if col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def classify_files(uploaded_files):
    """
    Intelligently classifies files based on their columns. This version is more flexible.
    """
    file_contents = {f.name: f.getvalue() for f in uploaded_files}
    probable_roles = {'timeseries': None, 'stations': []}
    
    for file in uploaded_files:
        try:
            temp_df = pd.read_csv(io.BytesIO(file_contents[file.name]), nrows=5)
            cols_cleaned = {col.lower().strip().replace('_', '').replace('-', '') for col in temp_df.columns}
            
            has_timestamp = any(ts_name in cols_cleaned for ts_name in ['timestamp', 'date', 'datetime'])
            has_lat = any(lat_name in cols_cleaned for lat_name in ['latitude', 'lat'])
            has_lon = any(lon_name in cols_cleaned for lon_name in ['longitude', 'lon', 'long'])
            
            if has_timestamp:
                probable_roles['timeseries'] = file.name
            elif has_lat and has_lon:
                probable_roles['stations'].append(file.name)
        except Exception as e:
            st.sidebar.warning(f"Could not read `{file.name}` during initial check. Skipping. Error: {e}")
            
    return probable_roles, file_contents

@st.cache_data
def get_processed_data(ts_data, filtered_stations, time_range_days=None):
    if filtered_stations.empty or ts_data is None:
        return pd.DataFrame()
    
    station_names_to_filter = filtered_stations['station_name'].unique()
    df_filtered = ts_data[ts_data['station_name'].isin(station_names_to_filter)].copy()
    
    if df_filtered.empty:
        return pd.DataFrame()

    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    agg_dict = {col: 'mean' for col in numeric_cols}
    
    if 'station_name' in df_filtered.columns:
        agg_dict['station_name'] = 'count'
        df_agg = df_filtered.groupby('timestamp').agg(agg_dict).rename(columns={'station_name': 'station_count'}).reset_index()
    else:
        df_agg = df_filtered.groupby('timestamp').agg(agg_dict).reset_index()

    if time_range_days:
        df_for_filtering = df_agg.set_index('timestamp')
        df_final = df_for_filtering.last(f'{time_range_days}D').reset_index()
    else:
        df_final = df_agg.copy()
        
    return df_final

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the earth (specified in decimal degrees).
    """
    R = 6371
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    distance = R * c
    return distance

def find_nearest_station(gw_station, rainfall_stations):
    """Finds the closest rainfall station to a given groundwater station using Haversine formula."""
    if rainfall_stations is None or rainfall_stations.empty: return None, 0
    lat1, lon1 = gw_station['latitude'], gw_station['longitude']
    
    distances = haversine(lat1, lon1, rainfall_stations['latitude'], rainfall_stations['longitude'])
    
    nearest_idx = distances.idxmin()
    nearest_station_info = rainfall_stations.loc[nearest_idx]
    return nearest_station_info, distances.min()

# --- NEW: Data Quality Reporting ---
@st.cache_data
def generate_data_quality_report(df, df_name):
    """Generates a markdown report on data quality for a given DataFrame."""
    report = f"### Data Quality: `{df_name}`\n"
    report += f"- **Shape**: {df.shape[0]} rows, {df.shape[1]} columns\n"
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_report = missing_percent[missing_percent > 0].sort_values(ascending=False)
    
    if not missing_report.empty:
        report += "- **Missing Values**:\n"
        for col, percent in missing_report.items():
            report += f"  - `{col}`: {percent:.1f}% missing\n"
    else:
        report += "- **Missing Values**: None found. ✅\n"
        
    if 'timestamp' in df.columns:
        if df['timestamp'].duplicated().any():
             report += f"- **Duplicate Timestamps**: Found {df['timestamp'].duplicated().sum()} duplicates. ⚠️\n"

    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
             report += f"- **Negative Values**: Found in `{col}`. Investigate if appropriate. ⚠️\n"
    return report

# --- MODIFIED: Policy Status Summary ---
@st.cache_data
def get_regional_status_summary(_ts_data, _gw_stations, group_by_col='state_name', critical_quantile=0.75):
    """Calculates regional groundwater status for the Policy tab with a configurable threshold."""
    if 'station_name' not in _ts_data.columns or 'station_name' not in _gw_stations.columns:
        return pd.DataFrame()
        
    merged_df = pd.merge(_ts_data, _gw_stations, on='station_name', how='left').dropna(subset=[group_by_col])

    def get_status(group):
        if group.empty or 'groundwaterlevel_mbgl' not in group.columns:
            return pd.Series({'status': 'No Data'})
        
        sorted_group = group.sort_values(by='timestamp')
        water_levels = sorted_group['groundwaterlevel_mbgl'].dropna()
        
        if len(water_levels) < 2:
            return pd.Series({'status': 'No Data'})
            
        latest_level = water_levels.iloc[-1]
        critical_level = water_levels.quantile(critical_quantile) # Use the configurable quantile
        return pd.Series({'status': 'Low/Critical' if latest_level > critical_level else 'Normal'})
            
    status_df = merged_df.groupby('station_name').apply(get_status).reset_index()
    
    # --- CORRECTED CODE ---
    # Use a set to ensure column names are unique before selecting and merging
    cols_to_merge = list(set(['station_name', group_by_col]))
    status_summary = pd.merge(status_df, _gw_stations[cols_to_merge], on='station_name', how='left')
    
    status_counts = status_summary.groupby([group_by_col, 'status']).size().reset_index(name='count')
    return status_counts

# --- NEW: Trend Calculation Function ---
@st.cache_data
def calculate_trend_stations(_ts_data):
    """Calculates long-term trends for all stations."""
    station_trends = {}
    for name, group in _ts_data.groupby('station_name'):
        df_station = group.dropna(subset=['groundwaterlevel_mbgl', 'timestamp']).sort_values('timestamp')
        if len(df_station) > 10: # Require a minimum number of data points
            df_station['time_ordinal'] = (df_station['timestamp'] - df_station['timestamp'].min()).dt.days
            X = df_station[['time_ordinal']]
            y = df_station['groundwaterlevel_mbgl']
            
            model = LinearRegression()
            model.fit(X, y)
            # Slope * 365 gives the approximate annual change in meters.
            # Positive slope is bad (deeper water level), negative is good.
            station_trends[name] = model.coef_[0] * 365 
            
    trends_df = pd.DataFrame(station_trends.items(), columns=['station_name', 'annual_trend_m'])
    return trends_df.sort_values('annual_trend_m', ascending=False)

# --- NEW: Monsoon Analysis Function ---
@st.cache_data
def analyze_monsoon_performance(_df_station):
    """Analyzes pre and post monsoon water levels for a single station."""
    df = _df_station.set_index('timestamp')
    df['year'] = df.index.year
    
    pre_monsoon = df[df.index.month.isin([3, 4, 5])].groupby('year')['groundwaterlevel_mbgl'].mean()
    post_monsoon = df[df.index.month.isin([10, 11, 12])].groupby('year')['groundwaterlevel_mbgl'].mean()
    
    monsoon_df = pd.DataFrame({'pre_monsoon_level': pre_monsoon, 'post_monsoon_level': post_monsoon})
    monsoon_df.dropna(inplace=True)
    monsoon_df['recharge_effect_m'] = monsoon_df['pre_monsoon_level'] - monsoon_df['post_monsoon_level']
    return monsoon_df.reset_index()

# --- NEW: Drought Detection Function ---
@st.cache_data
def detect_drought_events(_df_station, percentile_threshold=80):
    """Identifies historical drought periods based on a water level percentile."""
    if 'groundwaterlevel_mbgl' not in _df_station or _df_station['groundwaterlevel_mbgl'].dropna().empty:
        return []
        
    threshold = np.percentile(_df_station['groundwaterlevel_mbgl'].dropna(), percentile_threshold)
    df = _df_station[['timestamp', 'groundwaterlevel_mbgl']].copy()
    df['in_drought'] = df['groundwaterlevel_mbgl'] > threshold
    
    # Find blocks of consecutive drought days
    df['drought_block'] = (df['in_drought'].diff(1) != 0).astype('int').cumsum()
    
    drought_periods = []
    for block in df[df['in_drought']]['drought_block'].unique():
        drought_days = df[df['drought_block'] == block]
        start_date = drought_days['timestamp'].min().date()
        end_date = drought_days['timestamp'].max().date()
        duration = (end_date - start_date).days + 1
        peak_level = drought_days['groundwaterlevel_mbgl'].max()
        if duration > 30: # Only consider events longer than 30 days
            drought_periods.append({
                'Start': start_date, 'End': end_date, 'Duration (Days)': duration, f'Peak Level (mbgl)': peak_level
            })
    return drought_periods

# --- State Management Callback ---
def reset_forecast_state():
    """Resets session state variables related to the forecast."""
    st.session_state.forecast_generated = False
    if 'forecast_results' in st.session_state:
        st.session_state.forecast_results = {}
    if 'report_data' in st.session_state:
        del st.session_state['report_data']


# -------------------- Main App UI --------------------
st.title('💧 National Water Resources Intelligence Dashboard')

st.sidebar.header("Upload Your Data")
uploaded_files = st.sidebar.file_uploader(
    "Upload your 3 CSV files (GW Stations, RF Stations, Time-Series)", 
    type=['csv'], 
    accept_multiple_files=True
)

if len(uploaded_files) != 3:
    st.info("Welcome! Please upload your three required CSV files to begin.")
    st.stop()

probable_roles, file_contents = classify_files(uploaded_files)

if not probable_roles['timeseries'] or len(probable_roles['stations']) != 2:
    st.sidebar.error("Could not automatically identify the roles of all 3 files. Ensure one is a time-series (with a 'timestamp' column) and two are station files (with 'latitude'/'longitude' columns).")
    st.stop()
    
st.sidebar.markdown("---")

st.sidebar.info("Please confirm the file assignments:")
station_files_options = probable_roles['stations']
gw_station_fname = st.sidebar.selectbox("Select Groundwater Station File:", options=station_files_options, index=0)

default_rf_index = 1 if len(station_files_options) > 1 else 0
rf_station_fname = st.sidebar.selectbox("Select Rainfall Station File:", options=station_files_options, index=default_rf_index)

ts_fname = probable_roles['timeseries']

if gw_station_fname == rf_station_fname:
    st.sidebar.error("Groundwater and Rainfall station files cannot be the same. Please select different files.")
    st.stop()

st.sidebar.markdown("---")

# --- MODIFIED: Column Mapping section to handle both AI and Manual modes ---
with st.sidebar.expander("📝 Column Mapping", expanded=True):
    schemas = {
        'station': ['station_name', 'latitude', 'longitude', 'state_name', 'district_name', 'agency_name', 'basin'],
        'timeseries': ['station_name', 'timestamp', 'groundwaterlevel_mbgl', 'rainfall_mm', 'temperature_c', 'ph', 'turbidity_ntu', 'tds_ppm']
    }
    
    gw_raw_cols = pd.read_csv(io.BytesIO(file_contents[gw_station_fname]), nrows=0).columns.tolist()
    rf_raw_cols = pd.read_csv(io.BytesIO(file_contents[rf_station_fname]), nrows=0).columns.tolist()
    ts_raw_cols = pd.read_csv(io.BytesIO(file_contents[ts_fname]), nrows=0).columns.tolist()

    if st.session_state.get('ai_disabled', False):
        st.subheader("Manual Column Mapping")
        gw_mapping = manual_column_mapper(gw_station_fname, gw_raw_cols, schemas['station'])
        rf_mapping = manual_column_mapper(rf_station_fname, rf_raw_cols, schemas['station'])
        ts_mapping = manual_column_mapper(ts_fname, ts_raw_cols, schemas['timeseries'])
    # --- Corrected Code ---
    else:
        st.subheader("🤖 AI-Assisted Column Mapping")
        # Get the API key from session state ONCE before making the calls
        api_key = st.session_state.get("gemini_key")

        st.write(f"**`{gw_station_fname}` (as GW):**")
        gw_mapping = get_ai_column_mapping(gw_raw_cols, schemas['station'], api_key)
        if gw_mapping: st.json(gw_mapping, expanded=False)

        st.write(f"**`{rf_station_fname}` (as RF):**")
        rf_mapping = get_ai_column_mapping(rf_raw_cols, schemas['station'], api_key)
        if rf_mapping: st.json(rf_mapping, expanded=False)

        st.write(f"**`{ts_fname}` (as Time-Series):**")
        ts_mapping = get_ai_column_mapping(ts_raw_cols, schemas['timeseries'], api_key)
        if ts_mapping: st.json(ts_mapping, expanded=False)


if not all([gw_mapping, rf_mapping, ts_mapping]):
    st.error("Column mapping failed. Please map columns manually or check file headers/API key.")
    st.stop()

gw_stations = load_and_normalize_data(file_contents[gw_station_fname], gw_mapping)
rf_stations = load_and_normalize_data(file_contents[rf_station_fname], rf_mapping)
ts_data = load_and_normalize_data(file_contents[ts_fname], ts_mapping)

gw_stations['station_type'] = 'Groundwater'
rf_stations['station_type'] = 'Rainfall'

# --- MODIFIED: Filter station data to only include stations with corresponding time-series records ---
# This ensures that the dropdown menus and maps only show stations for which we have data to display.
if 'station_name' not in ts_data.columns:
    st.error("Fatal Error: Time-series file must contain a 'station_name' column after mapping.")
    st.stop()

# Create a single set of all station names that have time-series data.
ts_station_names = set(ts_data['station_name'].dropna().unique())

original_gw_count = len(gw_stations)
original_rf_count = len(rf_stations)

# Independently filter Groundwater stations: only keep stations that are found in the time-series file.
if 'station_name' in gw_stations.columns:
    gw_stations = gw_stations[gw_stations['station_name'].isin(ts_station_names)].copy()
else:
    st.error("Fatal Error: Groundwater station file must contain a 'station_name' column after mapping.")
    st.stop()
    
# Independently filter Rainfall stations: only keep stations that are found in the time-series file.
# Note: It is NOT required for a station to be in both the groundwater and rainfall station files.
if 'station_name' in rf_stations.columns:
    rf_stations = rf_stations[rf_stations['station_name'].isin(ts_station_names)].copy()
else:
    st.error("Fatal Error: Rainfall station file must contain a 'station_name' column after mapping.")
    st.stop()

st.sidebar.success("✔️ Files loaded and columns mapped successfully!")
st.sidebar.info(
    f"Found {len(gw_stations)} GW and {len(rf_stations)} RF stations with matching "
    f"time-series data.\n(Original files: {original_gw_count} GW, {original_rf_count} RF)"
)

if gw_stations.empty:
    st.error(
        "Fatal Error: No groundwater stations from your station file match any stations "
        "in your time-series file. Please check that station names are consistent across files."
    )
    st.stop()


# --- NEW: Data Quality Report in Sidebar ---
with st.sidebar.expander("📊 Automated Data Quality Report", expanded=True):
    st.markdown(generate_data_quality_report(gw_stations, gw_station_fname))
    st.markdown(generate_data_quality_report(rf_stations, rf_station_fname))
    st.markdown(generate_data_quality_report(ts_data, ts_fname))


# --- Sidebar Filtering ---
st.sidebar.header("Filter Stations")
ALL = "All"

# Create a unified, deduplicated list of all stations from both GW and RF files for populating the UI dropdowns.
# This ensures that if a station/district/state is in either file, it appears in the filter options.
all_stations_for_ui = pd.concat([
    gw_stations[['state_name', 'district_name', 'station_name', 'basin']],
    rf_stations[['state_name', 'district_name', 'station_name', 'basin']]
]).drop_duplicates().sort_values(by=['state_name', 'district_name', 'station_name']).reset_index(drop=True)

# --- Populate dropdowns using the unified list for a comprehensive selection ---
# The lists for the selectboxes are derived from the combined `all_stations_for_ui` dataframe.
state_list = [ALL] + sorted(all_stations_for_ui['state_name'].unique())
state = st.sidebar.selectbox("Select State", state_list, on_change=reset_forecast_state)

# Filter the UI options based on the selected state.
stations_in_ui_scope = all_stations_for_ui
if state != ALL:
    stations_in_ui_scope = stations_in_ui_scope[stations_in_ui_scope['state_name'] == state]

district_list = [ALL] + sorted(stations_in_ui_scope['district_name'].unique())
district = st.sidebar.selectbox("Select District", district_list, on_change=reset_forecast_state)

# Filter the UI options based on the selected district.
if district != ALL:
    stations_in_ui_scope = stations_in_ui_scope[stations_in_ui_scope['district_name'] == district]

basin_list = [ALL] + sorted(stations_in_ui_scope['basin'].dropna().unique())
basin = st.sidebar.selectbox("Select River Basin", basin_list, on_change=reset_forecast_state)

# Filter the UI options based on the selected basin.
if basin != ALL:
    stations_in_ui_scope = stations_in_ui_scope[stations_in_ui_scope['basin'] == basin]

station_list = [ALL] + sorted(stations_in_ui_scope['station_name'].unique())
gw_station_name = st.sidebar.selectbox("Select Groundwater Station", station_list, on_change=reset_forecast_state)

# --- Data Processing based on selection ---
# Now, filter the actual `gw_stations` dataframe based on the selections made.
# This ensures the rest of the app correctly processes only groundwater data as intended.
stations_in_scope = gw_stations
if state != ALL:
    stations_in_scope = stations_in_scope[stations_in_scope['state_name'] == state]
if district != ALL:
    stations_in_scope = stations_in_scope[stations_in_scope['district_name'] == district]
if basin != ALL:
    stations_in_scope = stations_in_scope[stations_in_scope['basin'] == basin]

single_station_mode = (gw_station_name != ALL)

if single_station_mode:
    filtered_gw_stations = stations_in_scope[stations_in_scope['station_name'] == gw_station_name]
else:
    filtered_gw_stations = stations_in_scope

header_location = "India"
if state != ALL: header_location = state
if district != ALL: header_location = f"{district}, {state}"
if basin != ALL: header_location = f"'{basin}' Basin"
if single_station_mode: header_location = gw_station_name

# --- Time Range Filter ---
st.sidebar.header("Filter Data by Time Range")
time_range_options = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 6 Months": 180, "Last Year": 365, "All Time": None}
selected_range_label = st.sidebar.selectbox("Select Time Range:", options=list(time_range_options.keys()), key='global_time_filter', on_change=reset_forecast_state)
days_to_filter = time_range_options[selected_range_label]

# --- Main Data Processing ---
df = get_processed_data(ts_data, filtered_gw_stations, days_to_filter)
df_unfiltered = get_processed_data(ts_data, filtered_gw_stations, time_range_days=None)

if df.empty:
    st.warning(f"No time-series data available for the current selection in '{selected_range_label}'. Please adjust your filters.")
    st.stop()

# --- Additional variables for single station mode ---
selected_gw_station, nearest_rf_station, distance = None, None, 0
if single_station_mode:
    selected_gw_station = filtered_gw_stations.iloc[0]
    nearest_rf_station, distance = find_nearest_station(selected_gw_station, rf_stations)


# --- MODIFIED: Navigation with State Management ---
tab_labels = [
    "🗺️ Unified Map View", 
    "📊 At-a-Glance Dashboard",
    "⚖️ Policy & Governance",
    "🏛️ Strategic Planning",
    "🔬 Research Hub", 
    "💧 Public Info", 
    "🔬 Advanced Hydrology",
    "📋 Generate Full Report"
]

# Initialize active_tab in session_state if it doesn't exist
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = tab_labels[0]

# Callback function to update the active tab
def set_active_tab():
    st.session_state.active_tab = st.session_state.navigation_radio

# Get the index of the active tab for the radio button
try:
    default_tab_index = tab_labels.index(st.session_state.active_tab)
except ValueError:
    default_tab_index = 0

st.radio(
    "Main navigation", 
    tab_labels, 
    index=default_tab_index,
    horizontal=True, 
    label_visibility="collapsed",
    key="navigation_radio", # Use a key for the widget
    on_change=set_active_tab # Use a callback to update state
)

selected_tab = st.session_state.active_tab


# --- Tab Content Rendering ---

# --- Unified Map View Tab ---
if selected_tab == tab_labels[0]:
    st.header(f"Water Monitoring Network in {header_location}")
    
    # --- NEW: Map View Controls ---
    map_col1, map_col2, map_col3 = st.columns([2,2,1])
    map_view_type = map_col1.radio("Select Map Style:", ["Points of Interest", "Heatmap (Density)"], horizontal=True)
    
    # --- NEW: Year-wise Filter ---
    available_years = sorted(ts_data['timestamp'].dt.year.unique(), reverse=True)
    selected_year = map_col2.selectbox("Analyze Geographic Status for Year:", ["All Time"] + available_years)

    filtered_rf_stations = rf_stations.copy()
    if state != ALL: filtered_rf_stations = filtered_rf_stations[filtered_rf_stations['state_name'] == state]
    if district != ALL: filtered_rf_stations = filtered_rf_stations[filtered_rf_stations['district_name'] == district]
    if basin != ALL: filtered_rf_stations = filtered_rf_stations[filtered_rf_stations['basin'] == basin]
    
    map_df_gw = filtered_gw_stations.copy()

    # --- NEW: Logic for Year-wise Status Coloring ---
    if selected_year != "All Time":
        ts_year_data = ts_data[ts_data['timestamp'].dt.year == selected_year]
        status_for_year = get_regional_status_summary(ts_year_data, gw_stations, group_by_col='station_name')
        status_map = status_for_year.set_index('station_name')['status'].to_dict()
        map_df_gw['status'] = map_df_gw['station_name'].map(status_map).fillna('No Data')
        color_map = {'Low/Critical': '#FF0000', 'Normal': '#0000FF', 'No Data': '#808080'} # Red/Blue/Gray for status
        map_df_gw['color'] = map_df_gw['status'].map(color_map)
        info_text = f"🔵 Normal | 🔴 Low/Critical in {selected_year}"
    else:
        map_df_gw['color'] = '#0000FF' # Default color
        info_text = "🔵 Groundwater Stations | 🟢 Rainfall Stations"

    map_df_rf = filtered_rf_stations
    map_df_rf['color'] = '#00FF00'
    map_df = pd.concat([map_df_gw, map_df_rf]).reset_index(drop=True)

    if map_view_type == "Heatmap (Density)":
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/dark-v9',
            initial_view_state=pdk.ViewState(
                latitude=map_df['latitude'].mean(),
                longitude=map_df['longitude'].mean(),
                zoom=6,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                   'HexagonLayer',
                   data=map_df[['latitude', 'longitude']],
                   get_position='[longitude, latitude]',
                   radius=5000,
                   elevation_scale=100,
                   elevation_range=[0, 1000],
                   pickable=True,
                   extruded=True,
                ),
            ],
        ))
    else: # Points of Interest
        map_df['size'] = 500
        if single_station_mode and selected_gw_station is not None:
            map_df.loc[map_df['station_name'] == gw_station_name, ['color', 'size']] = ['#FFD700', 1000] # Gold for selected
            if nearest_rf_station is not None:
                map_df.loc[map_df['station_name'] == nearest_rf_station['station_name'], ['color', 'size']] = ['#FFA500', 1000]
            info_text += f" | ⭐ Selected GW Station | 🟠 Nearest Rainfall Station ({distance:.2f} km away)"
        st.map(map_df, latitude='latitude', longitude='longitude', color='color', size='size')

    st.info(info_text)

# --- At-a-Glance Dashboard Tab ---
elif selected_tab == tab_labels[1]:
    st.header(f"At-a-Glance Dashboard for: {header_location}")
    st.subheader(f"Data for: {selected_range_label}")

    # --- Interactive Agency Filter ---
    st.markdown("---")
    st.markdown("#### Agency Drill-Down")
    agency_list = [ALL] + sorted(filtered_gw_stations['agency_name'].unique().tolist())
    selected_agency = st.selectbox(
        "Select an agency to filter the dashboard:",
        options=agency_list,
        help="Select an agency to see its specific metrics and highlight it on the chart."
    )
    
    # Create a copy to filter for this tab only
    tab_filtered_stations = filtered_gw_stations.copy()
    if selected_agency != ALL:
        tab_filtered_stations = tab_filtered_stations[tab_filtered_stations['agency_name'] == selected_agency]
        st.info(f"Dashboard filtered for agency: **{selected_agency}**")

    # Re-process data based on the potential new filter
    df_tab = get_processed_data(ts_data, tab_filtered_stations, days_to_filter)
    if df_tab.empty:
        st.warning(f"No time-series data available for '{selected_agency}' in the selected time range.")
        st.stop()
    
    if not single_station_mode:
        station_count = len(tab_filtered_stations)
        st.write(f"Displaying metrics for **{station_count}** station(s).")
    
    cols = st.columns(4)
    if 'groundwaterlevel_mbgl' in df_tab.columns and not df_tab['groundwaterlevel_mbgl'].dropna().empty:
        valid_gwl = df_tab['groundwaterlevel_mbgl'].dropna()
        if len(valid_gwl) > 1:
            delta_val = valid_gwl.iloc[-1] - valid_gwl.iloc[0]
            cols[1].metric("Most Recent Level (mbgl)", f"{valid_gwl.iloc[-1]:.2f} m", f"{delta_val:.2f} m change", "inverse")
        else:
            cols[1].metric("Most Recent Level (mbgl)", f"{valid_gwl.iloc[-1]:.2f} m")
        cols[0].metric("Avg Water Level (mbgl)", f"{valid_gwl.mean():.2f} m")
    
    if 'rainfall_mm' in df_tab.columns and not df_tab['rainfall_mm'].dropna().empty:
        cols[2].metric("Avg Rainfall", f"{df_tab['rainfall_mm'].mean():.2f} mm")
    if 'ph' in df_tab.columns and not df_tab['ph'].dropna().empty:
        cols[3].metric("Avg pH", f"{df_tab['ph'].mean():.2f}")
    
    cols2 = st.columns(4)
    if 'temperature_c' in df_tab.columns and not df_tab['temperature_c'].dropna().empty:
        cols2[0].metric("Avg Temperature", f"{df_tab['temperature_c'].mean():.1f} °C")
    if 'turbidity_ntu' in df_tab.columns and not df_tab['turbidity_ntu'].dropna().empty:
        cols2[1].metric("Latest Turbidity", f"{df_tab['turbidity_ntu'].dropna().iloc[-1]:.2f} NTU")
    if 'tds_ppm' in df_tab.columns and not df_tab['tds_ppm'].dropna().empty:
        cols2[2].metric("Latest TDS", f"{df_tab['tds_ppm'].dropna().iloc[-1]:.2f} ppm")

    st.markdown("---")
    st.markdown(f"#### Agency Contribution in {header_location}")
    agency_dist = filtered_gw_stations['agency_name'].value_counts()
    
    # --- Logic to "pull" the selected slice ---
    pull_values = [0.2 if agency == selected_agency else 0 for agency in agency_dist.index]

    # --- Custom Color Palette resembling the image ---
    custom_colors = ['#EF5350', '#66BB6A', '#29B6F6', '#FFEE58', '#AB47BC', '#FF7043'] # Red, Green, Light Blue, Yellow, Purple, Orange
    
    fig_pie = px.pie(agency_dist, values=agency_dist.values, names=agency_dist.index,
                     title=f'Groundwater Stations by Agency',
                     hole=0.4, # Creates the donut chart effect
                     template='plotly_dark',
                     color_discrete_sequence=custom_colors # Apply custom colors
                    )
    fig_pie.update_traces(
        pull=pull_values, # This is the key parameter for interaction
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=2)),
    )
    fig_pie.update_layout(
        legend_title_text='Agencies',
        title_font_size=20,
        legend_font_size=12
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Policy & Governance Tab ---
elif selected_tab == tab_labels[2]:
    st.header(f"Policy & Governance Dashboard")
    analysis_level = st.radio("Analyze Groundwater Stress by:", ("State", "River Basin"), horizontal=True)
    group_by_col = 'state_name' if analysis_level == "State" else 'basin'
    
    st.markdown(f"### 1. Regional Groundwater Stress Hotspots by {analysis_level}")
    # --- NEW: Configurable Threshold Slider ---
    critical_percentile = st.slider(
        "Define 'Critical' Level (Percentile of deepest historical levels):", 
        min_value=50, max_value=95, value=75, step=5,
        help="A station's latest reading is 'Critical' if it's deeper than this percentile of its all-time readings."
    )
    
    status_counts = get_regional_status_summary(
        ts_data, gw_stations, group_by_col=group_by_col, critical_quantile=critical_percentile / 100.0
    )
    
    if state != ALL and group_by_col == 'state_name':
        status_counts = status_counts[status_counts['state_name'] == state]
    if basin != ALL and group_by_col == 'basin':
        status_counts = status_counts[status_counts['basin'] == basin]
        
    if not status_counts.empty:
        fig_policy_status = px.bar(status_counts, x=group_by_col, y='count', color='status',
                                   title=f'Groundwater Status of Monitored Wells by {analysis_level}',
                                   labels={group_by_col: analysis_level, 'count': 'Number of Stations'},
                                   color_discrete_map={'Low/Critical': '#d9534f', 'Normal': '#5cb85c', 'No Data': '#777777'},
                                   template='plotly_dark')
        fig_policy_status.update_layout(bargap=0.2, legend_title_text='Status', yaxis_title="Number of Stations", xaxis_title=analysis_level)
        st.plotly_chart(fig_policy_status, use_container_width=True)
        
        # --- MODIFIED: Conditionally render AI button ---
        if not st.session_state.get('ai_disabled', False):
            if st.button("Get AI Policy Briefing", key='policy_briefing_ai'):
                prompt_data = status_counts.to_json(orient='records')
                prompt = f"""
                As a senior water resource policy advisor for the Government of India, analyze the following data summary of groundwater stress across different regions ({analysis_level}s).
                The data shows the number of monitoring stations categorized as 'Normal' or 'Low/Critical'.

                Data:
                {prompt_data}

                Based on this data, please provide a concise policy briefing that includes:
                1.  **Executive Summary**: A brief overview of the current situation.
                2.  **Key Hotspots**: Identify the top 2-3 most stressed regions ({analysis_level}s) that require immediate attention.
                3.  **Policy Recommendations**: Suggest 3 concrete, actionable policy recommendations to address the issues in the identified hotspots. Recommendations should be practical for implementation in the Indian context.
                4.  **Data Gaps**: Briefly mention if there are any potential data gaps or what additional information would be beneficial for a more detailed analysis.

                Format the response in clear, professional markdown.
                """
                analysis = get_gemini_analysis(prompt)
                if analysis:
                    st.markdown(analysis)
                else:
                    st.error("Failed to generate AI analysis. Please check your API key and try again.")
    else:
        st.warning(f"Not enough data to generate regional stress analysis for the selected {analysis_level}(s).")
    
    st.markdown("---")
    # --- NEW: Improving vs. Declining Stations ---
    st.markdown("### 2. Long-Term Station Trends")
    with st.spinner("Analyzing long-term trends for all stations..."):
        all_trends_df = calculate_trend_stations(ts_data)
        
    if not all_trends_df.empty:
        declining_df = all_trends_df[all_trends_df['annual_trend_m'] > 0].head(5)
        improving_df = all_trends_df[all_trends_df['annual_trend_m'] < 0].sort_values('annual_trend_m').head(5)
        
        col1, col2 = st.columns(2)
        with col1:
            st.error("Top 5 Declining Stations")
            st.dataframe(declining_df, column_config={
                "station_name": "Station Name",
                "annual_trend_m": st.column_config.NumberColumn("Annual Decline (m/year)", format="%.3f m")
            })
        with col2:
            st.success("Top 5 Improving Stations")
            st.dataframe(improving_df, column_config={
                "station_name": "Station Name",
                "annual_trend_m": st.column_config.NumberColumn("Annual Improvement (m/year)", format="%.3f m")
            })
    else:
        st.info("Insufficient long-term data to calculate station trends.")

    st.markdown("---")
    # --- MODIFIED: Conditionally render AI feature ---
    if not st.session_state.get('ai_disabled', False):
        st.markdown("### 3. AI Policy Simulator & Advisor")
        policy_goal = st.selectbox("Select Policy Goal:", ["Increase Groundwater Recharge", "Reduce Water Depletion", "Improve Drought Resilience", "Ensure Equitable Water Access", "Modernize Water Governance"])
        if st.button(f"Generate Policy Brief for '{policy_goal}'", key='policy_brief_ai'):
            prompt = f"""
            As a water policy expert specializing in Indian water governance, create a detailed policy brief on the topic: **"{policy_goal}"**.

            The brief should be structured as follows:
            1.  **Introduction**: Briefly explain the importance of this policy goal in the context of India's water security challenges.
            2.  **Key Challenges**: Outline 2-3 major challenges in achieving this goal in India.
            3.  **Strategic Interventions**: Propose a set of 3-5 strategic interventions or policy measures. For each intervention, provide a brief description of what it entails and its potential impact.
            4.  **Implementation Roadmap**: Suggest key steps for implementing these strategies, including the roles of central government, state governments, and local bodies.
            5.  **Conclusion**: A short concluding paragraph summarizing the importance of action.

            Your response should be well-structured, informative, and formatted in professional markdown.
            """
            analysis = get_gemini_analysis(prompt)
            if analysis:
                st.markdown(analysis)
            else:
                st.error("Failed to generate AI analysis. Please check your API key and try again.")


# --- Strategic Planning Tab ---
elif selected_tab == tab_labels[3]:
    st.header(f"Strategic Planning Dashboard for: {header_location}")
    if single_station_mode and 'groundwaterlevel_mbgl' in df_unfiltered.columns and not df_unfiltered.empty:
        st.info("This dashboard uses the complete historical dataset for long-term strategic analysis.")
        st.markdown("---")
        st.markdown("### 1. Long-Term Historical Water Level Trend")
        
        fig_long_term = px.line(df_unfiltered, x='timestamp', y='groundwaterlevel_mbgl', 
                                title='Complete History of Water Levels', template='plotly_dark', markers=True)
        fig_long_term.update_traces(marker=dict(size=5))
        fig_long_term.update_layout(yaxis_title='Groundwater Level (mbgl)', xaxis_title='Date')
        st.plotly_chart(fig_long_term, use_container_width=True)

        # --- MODIFIED: Conditionally render AI button ---
        if not st.session_state.get('ai_disabled', False):
            if st.button("Get AI Trend Analysis for Planning", key='planning_analysis_ai'):
                prompt_data = df_unfiltered[['timestamp', 'groundwaterlevel_mbgl']].tail(100).to_string()
                prompt = f"""
                As a senior hydrologist preparing a report for strategic planning, analyze the following long-term historical water level data for the station: **{gw_station_name}**.
                The 'groundwaterlevel_mbgl' is meters below ground level (a higher number means a lower water level).

                Recent Data Sample:
                {prompt_data}

                Based on the overall trend (not just this sample), provide an analysis covering:
                1.  **Overall Trend Identification**: Describe the primary long-term trend (e.g., stable, declining, recharging, cyclical).
                2.  **Seasonal Patterns**: Comment on any observable seasonal variations (e.g., impact of monsoon).
                3.  **Implications for Planning**: What are the key strategic implications of this trend? For example, if the trend is declining, what does this mean for future water availability and the need for intervention?

                Keep the analysis concise and focused on providing actionable insights for a water resource planner.
                """
                analysis = get_gemini_analysis(prompt)
                if analysis:
                    st.markdown(analysis)
                else:
                    st.error("Failed to generate AI analysis. Please check your API key and try again.")

        st.markdown("---")
        st.markdown("### 2. Sustainable Yield Estimation")
        sy_planning = st.number_input("Enter Specific Yield (Sy) for Planning:", 0.01, 0.50, 0.15, 0.01, key='sy_planning')
        
        planning_df = df_unfiltered.set_index('timestamp').asfreq('D').interpolate()
        
        planning_df['gw_level_change'] = planning_df['groundwaterlevel_mbgl'].diff()
        planning_df['recharge_mm'] = planning_df.apply(lambda row: (row['gw_level_change'] * -1 * sy_planning * 1000) if row['gw_level_change'] < 0 else 0, axis=1)

        avg_annual_recharge = planning_df['recharge_mm'].resample('Y').sum().mean()
        sustainable_yield_mm = avg_annual_recharge * 0.7 
        st.metric("Average Estimated Annual Recharge", f"{avg_annual_recharge:.2f} mm/year")
        st.metric("Estimated Sustainable Yield", f"{sustainable_yield_mm:.2f} mm/year", help="Calculated as 70% of recharge, a common heuristic for sustainable extraction.")
        
        st.markdown("---")
        st.markdown("### 3. Supply vs. Demand Scenario Modeling")
        area_influence = st.number_input("Enter Area of Influence (sq. km):", 0.1, value=10.0, step=1.0, key='area_planning')
        sustainable_volume_m3 = (sustainable_yield_mm / 1000) * (area_influence * 1_000_000)
        projected_demand_m3_day = st.number_input("Enter Projected Daily Demand (m³):", 0, value=1000, step=100)
        projected_demand_m3_year = projected_demand_m3_day * 365
        st.write(f"**Estimated Sustainable Supply:** {sustainable_volume_m3:,.0f} m³/year")
        st.write(f"**Projected Annual Demand:** {projected_demand_m3_year:,.0f} m³/year")
        if projected_demand_m3_year > sustainable_volume_m3:
            st.error(f"Deficit: {projected_demand_m3_year - sustainable_volume_m3:,.0f} m³/year.")
        else:
            st.success(f"Surplus: {sustainable_volume_m3 - projected_demand_m3_year:,.0f} m³/year.")
        
        # --- MODIFIED: Conditionally render AI button ---
        if not st.session_state.get('ai_disabled', False):
            if st.button("Generate Strategic Recommendations", key='strategic_ai'):
                deficit_status = f"There is a projected annual deficit of {projected_demand_m3_year - sustainable_volume_m3:,.0f} m³." if projected_demand_m3_year > sustainable_volume_m3 else f"There is a projected annual surplus of {sustainable_volume_m3 - projected_demand_m3_year:,.0f} m³."
                prompt = f"""
                As a strategic consultant for a regional water authority, you are presented with the following scenario for an area of {area_influence} sq. km around station {gw_station_name}:
                - Estimated Sustainable Annual Supply: {sustainable_volume_m3:,.0f} m³
                - Projected Annual Demand: {projected_demand_m3_year:,.0f} m³
                - Result: {deficit_status}

                Based on this supply-demand scenario, provide a set of strategic recommendations.
                If there is a deficit, focus on:
                1.  **Demand-Side Management**: Suggest 2-3 measures to reduce water consumption (e.g., for agriculture, domestic use).
                2.  **Supply-Side Augmentation**: Suggest 2-3 measures to increase water availability (e.g., rainwater harvesting, wastewater treatment, new infrastructure).
                
                If there is a surplus, focus on:
                1.  **Sustainable Allocation**: How can the surplus be allocated for economic growth while ensuring long-term sustainability?
                2.  **Resilience Building**: How can the surplus be used to build resilience against future droughts or climate change?

                Provide clear, actionable recommendations in a markdown format.
                """
                analysis = get_gemini_analysis(prompt)
                if analysis:
                    st.markdown(analysis)
                else:
                    st.error("Failed to generate AI analysis. Please check your API key and try again.")
    else:
        st.info("Please select a single station to access detailed planning tools like Sustainable Yield and Scenario Modeling.")

# --- Research Hub Tab ---
elif selected_tab == tab_labels[4]:
    st.header(f"Research Hub for: {header_location}")
    st.subheader(f"Data for: {selected_range_label}")
    if not single_station_mode:
        station_count_research = int(df['station_count'].max()) if 'station_count' in df.columns else len(filtered_gw_stations)
        st.info(f"Displaying **average** trends for **{station_count_research}** stations.")
        
    st.markdown("#### Comprehensive Water Quality Analysis")
    fig_quality = make_subplots(specs=[[{"secondary_y": True}]])
    if 'ph' in df.columns: fig_quality.add_trace(go.Scatter(x=df['timestamp'], y=df['ph'], name='pH'), secondary_y=False)
    if 'tds_ppm' in df.columns: fig_quality.add_trace(go.Scatter(x=df['timestamp'], y=df['tds_ppm'], name='TDS (ppm)'), secondary_y=True)
    if 'turbidity_ntu' in df.columns: fig_quality.add_trace(go.Scatter(x=df['timestamp'], y=df['turbidity_ntu'], name='Turbidity (NTU)'), secondary_y=True)
    
    title_suffix = " (Regional Average)" if not single_station_mode else ""
    fig_quality.update_layout(title_text=f"Water Quality Over Time{title_suffix}", 
                              template='plotly_dark', legend_title_text='Parameters')
    fig_quality.update_yaxes(title_text="pH Level", secondary_y=False)
    fig_quality.update_yaxes(title_text="TDS (ppm) / Turbidity (NTU)", secondary_y=True)
    st.plotly_chart(fig_quality, use_container_width=True)
    
    st.markdown("---")
    # --- MODIFIED: Conditionally render AI feature ---
    if not st.session_state.get('ai_disabled', False):
        st.markdown("#### 🤖 AI Correlation Analyst")
        st.write("Select two parameters to analyze their relationship.")
        cols_for_corr = [col for col in ['groundwaterlevel_mbgl', 'rainfall_mm', 'temperature_c', 'ph', 'turbidity_ntu', 'tds_ppm'] if col in df.columns]
        col1, col2 = st.columns(2)
        param1 = col1.selectbox("Parameter 1:", cols_for_corr, index=0)
        param2 = col2.selectbox("Parameter 2:", cols_for_corr, index=1 if len(cols_for_corr) > 1 else 0)
        if st.button("Analyze Correlation with AI", key="corr_ai"):
            corr_df = df[[param1, param2]].dropna()
            if len(corr_df) > 2:
                correlation = corr_df[param1].corr(corr_df[param2])
                prompt_data = corr_df.tail(50).to_string()
                prompt = f"""
                As a research hydrologist, analyze the relationship between two parameters: '{param1}' and '{param2}'.
                The calculated Pearson correlation coefficient is: **{correlation:.2f}**.

                Here is a sample of the recent data:
                {prompt_data}

                Provide a scientific interpretation of this relationship:
                1.  **Explain the Correlation**: Based on the correlation coefficient, describe the strength and direction of the relationship (e.g., strong positive, weak negative, no correlation).
                2.  **Hydrological Context**: Explain the likely physical or chemical reasons for this relationship in a water resource system. For example, why would rainfall be correlated with groundwater level?
                3.  **Implications for Research**: What might this relationship imply for water quality management or resource monitoring?

                Format the response as a concise analysis.
                """
                analysis = get_gemini_analysis(prompt)
                if analysis:
                    st.markdown(f"### AI-Powered Correlation Analysis: `{param1}` vs. `{param2}`")
                    st.markdown(analysis)
                else:
                    st.error("Failed to generate AI analysis. Please check your API key and try again.")
            else:
                st.warning("Not enough overlapping data points to analyze the correlation.")
        st.markdown("---")

    st.markdown("#### 💧 High-Accuracy Predictive Forecast")
    if single_station_mode and 'groundwaterlevel_mbgl' in df_unfiltered.columns:
        if 'forecast_generated' not in st.session_state:
            st.session_state.forecast_generated = False
        if 'forecast_results' not in st.session_state:
            st.session_state.forecast_results = {}

        forecast_df = df_unfiltered[['timestamp', 'groundwaterlevel_mbgl', 'rainfall_mm']].copy().set_index('timestamp').asfreq('D').interpolate()
        forecast_days = st.slider("Days to forecast:", 7, 90, 30, key='forecast_slider_research')
        
        st.warning("Note: The SARIMAX model below uses fixed parameters. For best results, these should be tuned for each specific dataset (e.g., using `pmdarima.auto_arima`).")
        
        if st.button("Generate Accurate Forecast", key='forecast_button_research'):
            with st.spinner(f"Running SARIMAX model for {forecast_days} days..."):
                try:
                    endog, exog = forecast_df['groundwaterlevel_mbgl'], forecast_df['rainfall_mm']
                    
                    order_params = (1, 1, 1)
                    seasonal_order_params = (1, 1, 1, 12)
                    
                    model_fit = SARIMAX(endog, exog=exog, order=order_params, seasonal_order=seasonal_order_params).fit(disp=False)
                    future_exog = pd.DataFrame({'rainfall_mm': [exog.mean()] * forecast_days}, index=pd.date_range(start=endog.index[-1] + pd.Timedelta(days=1), periods=forecast_days))
                    forecast_obj = model_fit.get_forecast(steps=forecast_days, exog=future_exog)
                    forecast_ci, forecast_values = forecast_obj.conf_int(), forecast_obj.predicted_mean
                    st.success("✅ Forecast Generated Successfully!")
                    
                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(x=forecast_ci.index.tolist() + forecast_ci.index.tolist()[::-1], y=forecast_ci.iloc[:, 1].tolist() + forecast_ci.iloc[:, 0].tolist()[::-1], fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval'))
                    fig_forecast.add_trace(go.Scatter(x=df['timestamp'], y=df['groundwaterlevel_mbgl'], mode='lines', name=f'Historical Data ({selected_range_label})', line=dict(color='cyan')))
                    fig_forecast.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values, mode='lines', name='Forecasted Water Level', line=dict(color='orange', dash='dot')))
                    fig_forecast.update_layout(title="Water Level Forecast vs. Historical Data", 
                                               yaxis_title="Groundwater Level (mbgl)",
                                               template='plotly_dark', legend_title_text='Data Series')
                    
                    st.session_state.forecast_results = {
                        'fig': fig_forecast, 'endog': endog,
                        'forecast_values': forecast_values, 'forecast_ci': forecast_ci,
                        'forecast_days': forecast_days, 'order': order_params, 
                        'seasonal_order': seasonal_order_params
                    }
                    st.session_state.forecast_generated = True
                    # st.rerun() # We remove the rerun to prevent the tab switch feeling

                except Exception as e:
                    st.error(f"An error occurred during forecasting: {e}")
                    st.session_state.forecast_generated = False

        # --- MODIFIED: More robust check for forecast results ---
        if st.session_state.get('forecast_generated', False) and 'fig' in st.session_state.get('forecast_results', {}):
            results = st.session_state.forecast_results
            st.plotly_chart(results['fig'], use_container_width=True)

            st.markdown("#### Forecast Data")
            forecast_table = pd.DataFrame({
                'Predicted Level (mbgl)': results['forecast_values'],
                'Lower Confidence (mbgl)': results['forecast_ci'].iloc[:, 0],
                'Upper Confidence (mbgl)': results['forecast_ci'].iloc[:, 1]
            })
            st.dataframe(forecast_table.style.format("{:.2f}"))

            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv().encode('utf-8')

            csv = convert_df_to_csv(forecast_table)
            st.download_button(
                label="📥 Export Forecast as CSV",
                data=csv,
                file_name=f"{gw_station_name}_forecast.csv",
                mime='text/csv',
            )
            # --- MODIFIED: Conditionally render AI button ---
            if not st.session_state.get('ai_disabled', False):
                if st.button("Get AI Research Analysis of Forecast", key='forecast_analysis_ai'):
                    results = st.session_state.forecast_results
                    prompt_data = results['endog'].tail(30).to_string()
                    forecast_summary = results['forecast_values'].to_string()
                    prompt = f"""
                    As a research scientist, analyze the following time-series forecast for groundwater level (mbgl) at station {gw_station_name}.
                    
                    **Recent Historical Data:**
                    {prompt_data}

                    **Forecasted Values for the next {results['forecast_days']} days:**
                    {forecast_summary}

                    The forecast was generated using a SARIMAX model with order={results['order']} and seasonal_order={results['seasonal_order']}. A confidence interval was also generated.

                    Provide a research-oriented analysis:
                    1.  **Forecast Interpretation**: What does the forecast predict (e.g., a continued decline, a seasonal recovery, stabilization)?
                    2.  **Confidence Assessment**: What does the confidence interval imply about the certainty of the forecast?
                    3.  **Model Implications**: Briefly comment on what the SARIMAX model parameters might suggest about the underlying data's properties (e.g., seasonality, trend).
                    4.  **Further Research**: Suggest two potential research questions or avenues for further investigation based on these results.
                    """
                    analysis = get_gemini_analysis(prompt)
                    if analysis:
                        st.session_state.forecast_results['analysis'] = analysis
                        # st.rerun() # We remove the rerun to prevent the tab switch feeling
                    else:
                        st.error("Failed to generate AI analysis. Please check your API key and try again.")
                
                if 'analysis' in st.session_state.forecast_results:
                    st.markdown(st.session_state.forecast_results['analysis'])

    else:
        st.info("Predictive forecasting requires a consistent time-series from a single source. Please select a single station.")

# --- Public Info Tab ---
elif selected_tab == tab_labels[5]:
    st.header(f"Public Water Information for {header_location}")
    st.subheader(f"Data for: {selected_range_label}")
    if 'groundwaterlevel_mbgl' in df.columns:
        if not single_station_mode:
            station_count_public = int(df['station_count'].max()) if 'station_count' in df.columns else len(filtered_gw_stations)
            st.info(f"Displaying regional average status for **{station_count_public}** stations.")
            
        latest = df.iloc[-1]
        st.markdown("#### Current Water Status")
        
        min_level = df['groundwaterlevel_mbgl'].min()
        max_level = df['groundwaterlevel_mbgl'].max()
        avg_level = df['groundwaterlevel_mbgl'].mean()
        latest_level_val = latest['groundwaterlevel_mbgl']
        title_text = "Current GW Level (mbgl)" if single_station_mode else "Current Regional Avg. GW Level (mbgl)"

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = latest_level_val,
            title = {'text': title_text, 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [min_level * 0.9, max_level * 1.1], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#277DA1"},
                'steps' : [
                    {'range': [min_level * 0.9, avg_level], 'color': "#57C4E5"},
                    {'range': [avg_level, max_level * 1.1], 'color': "#F94144"}
                ],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.9, 'value': avg_level}
            }))
        fig_gauge.update_layout(margin=dict(l=20, r=20, t=50, b=20), template='plotly_dark')
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.info("Lighter blue indicates water levels are better (lower mbgl) than the period average; Lighter red indicates they are worse (higher mbgl). The red line marks the average.")
        
        # --- MODIFIED: Conditionally render AI feature ---
        if not st.session_state.get('ai_disabled', False):
            st.markdown("#### 🤖 AI Summary for You")
            if st.button("Get a Simple Summary", key='summary_button'):
                latest_level = df['groundwaterlevel_mbgl'].dropna().iloc[-1]
                avg_level = df['groundwaterlevel_mbgl'].mean()
                prompt = f"""
                You are a helpful assistant for a public water information portal. Your goal is to explain the water situation in simple, clear language that anyone can understand.

                Here is the current data for {header_location}:
                - The most recent average groundwater level is: {latest_level:.2f} meters below the ground.
                - The average level over the last period ({selected_range_label}) was: {avg_level:.2f} meters below the ground.
                - A HIGHER number means the water is DEEPER and thus less available.

                Please provide a one-paragraph summary explaining what this means.
                - Start by stating the current situation (e.g., "Currently, the water level is...").
                - Compare the current level to the recent average. Is it better or worse than usual?
                - End with a simple concluding sentence about the importance of using water wisely.
                - Do not use technical jargon. Keep it very simple.
                """
                analysis = get_gemini_analysis(prompt)
                if analysis:
                    st.info(analysis)
                else:
                    st.error("Failed to generate summary. Please check your API key and try again.")
    else:
        st.warning("No groundwater level data available to display for this selection.")

# --- Advanced Hydrology Tab ---
elif selected_tab == tab_labels[6]:
    st.header(f"Advanced Hydrological Analysis for: {header_location}")
    if 'groundwaterlevel_mbgl' not in df_unfiltered.columns or df_unfiltered['groundwaterlevel_mbgl'].dropna().empty:
        st.warning("This tab requires 'groundwaterlevel_mbgl' data, which is not available for this selection.")
        st.stop()
    
    if not single_station_mode:
        st.info("Advanced hydrology tools work best with a single station. Displaying aggregated regional analysis.")

    hydro_df = df.copy().set_index('timestamp').asfreq('D').interpolate()
    
    st.markdown("### 1. Water Level Fluctuation & Volatility")
    if len(hydro_df) > 90:
        hydro_df['90_day_avg'] = hydro_df['groundwaterlevel_mbgl'].rolling(window=90).mean()
        hydro_df['volatility'] = hydro_df['groundwaterlevel_mbgl'].rolling(window=90).std()

        fig_fluctuation = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces
        fig_fluctuation.add_trace(go.Scatter(x=hydro_df.index, y=hydro_df['groundwaterlevel_mbgl'], name='Daily Water Level', line=dict(color='skyblue')), secondary_y=False)
        fig_fluctuation.add_trace(go.Scatter(x=hydro_df.index, y=hydro_df['90_day_avg'], name='90-Day Avg. Trend', line=dict(color='orange', dash='dot')), secondary_y=False)
        fig_fluctuation.add_trace(go.Scatter(x=hydro_df.index, y=hydro_df['volatility'], name='90-Day Volatility', line=dict(color='lightgreen')), secondary_y=True)

        # Set titles and labels
        fig_fluctuation.update_layout(title_text="Water Level Fluctuation and Volatility", template='plotly_dark')
        fig_fluctuation.update_yaxes(title_text="Groundwater Level (mbgl)", secondary_y=False)
        fig_fluctuation.update_yaxes(title_text="Volatility (Std. Dev.)", secondary_y=True)
        
        st.plotly_chart(fig_fluctuation, use_container_width=True)
        # --- MODIFIED: Conditionally render AI button ---
        if not st.session_state.get('ai_disabled', False):
            if st.button("Get AI Fluctuation Analysis", key="fluctuation_ai"):
                prompt_data = hydro_df.tail(30).to_string() # Use recent data for prompt
                prompt = f"""
                Analyze the following recent water level data for station {gw_station_name}.
                The data includes daily water level (mbgl), a 90-day average trendline, and 90-day volatility (standard deviation).
                
                Data snippet:
                {prompt_data}

                Based on this data, provide a brief analysis covering:
                1. The current trend of the water level compared to its 90-day average.
                2. The level of volatility. Is the water level stable or fluctuating significantly?
                3. Any potential implications (e.g., signs of rapid depletion, recharge, or seasonal effects).
                Keep the analysis concise and easy for a water resource manager to understand.
                """
                analysis = get_gemini_analysis(prompt)
                if analysis:
                    st.markdown(analysis)
                else:
                    st.error("Failed to generate AI analysis. Please check your API key and try again.")
    else:
        st.info("At least 90 days of data are required to calculate fluctuation and volatility.")
    st.markdown("---")

    st.markdown("### 2. Smoothed Trend Analysis (EWMA)")
    ewma_span = st.slider("Select EWMA Span (days):", 7, 180, 30, key='ewma_span')
    hydro_df['ewma'] = hydro_df['groundwaterlevel_mbgl'].ewm(span=ewma_span, adjust=False).mean()
    fig_ewma = go.Figure()
    fig_ewma.add_trace(go.Scatter(x=hydro_df.index, y=hydro_df['groundwaterlevel_mbgl'], mode='lines', name='Daily GW Level', opacity=0.5, line=dict(color='cyan', width=1)))
    fig_ewma.add_trace(go.Scatter(x=hydro_df.index, y=hydro_df['ewma'], mode='lines', name=f'{ewma_span}-Day EWMA Trend', line=dict(color='yellow', width=3)))
    fig_ewma.update_layout(title="Exponentially Weighted Moving Average Trend", yaxis_title="Groundwater Level (mbgl)", template='plotly_dark')
    st.plotly_chart(fig_ewma, use_container_width=True)
    st.markdown("---")
    
    if single_station_mode:
        st.markdown(f"### 3. Seasonal Aquifer Performance for '{gw_station_name}'")
        monsoon_data = analyze_monsoon_performance(df_unfiltered)
        if not monsoon_data.empty:
            fig_monsoon = go.Figure()
            fig_monsoon.add_trace(go.Bar(x=monsoon_data['year'], y=monsoon_data['pre_monsoon_level'], name='Pre-Monsoon Level', marker_color='brown'))
            fig_monsoon.add_trace(go.Bar(x=monsoon_data['year'], y=monsoon_data['post_monsoon_level'], name='Post-Monsoon Level', marker_color='blue'))
            fig_monsoon.update_layout(barmode='group', title='Pre vs. Post Monsoon Water Levels', yaxis_title='GW Level (mbgl)', template='plotly_dark')
            st.plotly_chart(fig_monsoon, use_container_width=True)
            
            st.metric("Average Monsoon Recharge Effect", f"{monsoon_data['recharge_effect_m'].mean():.2f} m")
            st.dataframe(monsoon_data)
        else:
            st.info("Insufficient data across pre and post-monsoon seasons to perform analysis.")
        st.markdown("---")

        st.markdown(f"### 4. Historical Drought Event Analysis for '{gw_station_name}'")
        drought_percentile = st.slider("Define Drought Threshold (Percentile)", 70, 99, 85)
        drought_events = detect_drought_events(df_unfiltered, drought_percentile)
        if drought_events:
            st.warning(f"Detected {len(drought_events)} significant drought periods (water level > {drought_percentile}th percentile).")
            st.dataframe(pd.DataFrame(drought_events))
        else:
            st.success("No significant drought periods detected based on the current threshold.")
    else:
        st.info("Please select a single station for Seasonal Performance and Drought Event analysis.")
        
# --- NEW: Generate Full Report Tab ---
elif selected_tab == tab_labels[7]:
    st.header(f"📋 Consolidated Intelligence Report")
    
    if st.button("➕ Generate Full Report for Current Selection"):
        report = {
            "header": f"Water Intelligence Report for: {header_location}",
            "filters": {
                "State": state, "District": district, "River Basin": basin,
                "Station": gw_station_name if single_station_mode else f"{len(filtered_gw_stations)} stations",
                "Time Period": selected_range_label
            },
            "summary_metrics": {},
            "policy_insights": {},
            "long_term_trend": "N/A (Select single station)",
            "forecast": "N/A (Generate on Research Hub tab first)"
        }
        
        # At-a-Glance
        if 'groundwaterlevel_mbgl' in df.columns and not df['groundwaterlevel_mbgl'].dropna().empty:
            report["summary_metrics"]["Avg Water Level (mbgl)"] = f"{df['groundwaterlevel_mbgl'].mean():.2f} m"
            report["summary_metrics"]["Most Recent Level (mbgl)"] = f"{df['groundwaterlevel_mbgl'].dropna().iloc[-1]:.2f} m"
        if 'rainfall_mm' in df.columns and not df['rainfall_mm'].dropna().empty:
            report["summary_metrics"]["Avg Rainfall"] = f"{df['rainfall_mm'].mean():.2f} mm"
        
        # Policy Insights
        status_counts = get_regional_status_summary(ts_data, gw_stations, group_by_col='state_name')
        if not status_counts.empty:
            critical_df = status_counts[status_counts['status'] == 'Low/Critical']
            if not critical_df.empty:
                top_stressed_state = critical_df.sort_values('count', ascending=False).iloc[0]['state_name']
                report["policy_insights"]["Top Stressed State"] = top_stressed_state
        
        # Long-term trend for single station
        if single_station_mode:
            trends = calculate_trend_stations(ts_data[ts_data['station_name'] == gw_station_name])
            if not trends.empty:
                trend_val = trends.iloc[0]['annual_trend_m']
                report["long_term_trend"] = f"{trend_val:.3f} m/year change"
        
        # Forecast
        if st.session_state.get('forecast_generated', False):
            results = st.session_state.forecast_results
            forecast_end_val = results['forecast_values'].iloc[-1]
            report["forecast"] = f"Next {results['forecast_days']} days: Forecasted to reach {forecast_end_val:.2f} mbgl"
            
        st.session_state['report_data'] = report

    if 'report_data' in st.session_state:
        report_data = st.session_state['report_data']
        st.subheader(report_data['header'])
        st.markdown("---")
        
        st.markdown("#### Selection Criteria")
        st.json(report_data['filters'])
        
        st.markdown("#### Summary Metrics")
        st.json(report_data['summary_metrics'])
        
        st.markdown("#### Key Insights")
        col1, col2 = st.columns(2)
        col1.metric("Top Stressed State", report_data.get("policy_insights", {}).get("Top Stressed State", "N/A"))
        col2.metric("Long-Term Trend", report_data["long_term_trend"])
        
        st.metric("Short-Term Forecast", report_data["forecast"])
        st.markdown("---")

        # Download button
        report_json = json.dumps(report_data, indent=2)
        st.download_button(
            label="📥 Download Report Data (JSON)",
            data=report_json,
            file_name=f"water_report_{header_location.replace(' ', '_').lower()}.json",
            mime="application/json"
        )


