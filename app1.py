import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import requests
import pytz
from datetime import datetime
import joblib

# Load model and label encoders
model = joblib.load("energy_demand_xgb_model.joblib")
source_type_encoder = joblib.load("source_type_encoder.joblib")
region_encoder = joblib.load("region_encoder.joblib")

API_KEY = 'a93063961d6ba847cdae4c0a59ec93c4'

st.set_page_config(page_title="Energy Demand Dashboard", layout="wide")

primary_color = "#1E81B0"
background_color = "#EAF6F9"
text_color = "#034F84"

st.markdown(f"""
    <h1 style="text-align:center; color:#003366; font-size:40px;">
        Smart Grid Analytics Dashboard
    </h1>
<style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .css-1d391kg {{
        background-color: {primary_color};
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_excel('PROJECT_DATASET.xlsx')
    df['datetime_iso'] = pd.to_datetime(df['datetime_iso'], errors='coerce')
    df = df.sort_values('datetime_iso').reset_index(drop=True)
    return df

def now_ist():
    return datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%d-%m-%Y %H:%M:%S")

def fetch_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    r = requests.get(url)
    if r.status_code==200:
        data = r.json()
        return {
            'temperature_c': data['main']['temp'],
            'humidity_percent': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed_kmph': data['wind']['speed'] * 3.6,
            'clouds_percent': data['clouds']['all'],
            'rainfall_mm': data.get('rain', {}).get('1h', 0.0),
            'weather_desc': data['weather'][0]['description'],
        }
    else:
        return None


def predict_demand(df_features):
    features = [
        'new_residential_units', 'commercial_space_added_sqm', 'population_growth',
        'real_estate_index', 'wind_generation_mw', 'installed_capacity_mw', 'is_holiday',
        'temperature_c', 'humidity_percent', 'radiation_wm2', 'wind_speed_kmph',
        'rainfall_mm', 'power_purchased_mw', 'cost_per_mw', 'industrial_load_mw',
        'commercial_load_mw', 'residential_load_mw', 'solar_generation_mw',
        'solar_irradiance', 'reserve_generation_mw', 'grid_balancing_mw',
        'source_type_encoded', 'region_encoded', 'peak_demand_flag'
    ]
    X = xgb.DMatrix(df_features[features])
    preds = model.predict(X)
    return preds

def optimize_grid(demand, storage_capacity):
    return np.minimum(demand, storage_capacity)

def calculate_cost_savings(saved_kwh, num_houses, cost_per_kwh):
    return saved_kwh / num_houses * cost_per_kwh

# Load data
df = load_data()

# Sidebar input controls
st.sidebar.header("Dashboard Inputs")

city = st.sidebar.text_input("City", "Mumbai")
start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    value=[df['datetime_iso'].min().date(), df['datetime_iso'].max().date()],
    min_value=df['datetime_iso'].min().date(),
    max_value=df['datetime_iso'].max().date()
)

storage_capacity = st.sidebar.slider("Storage Capacity (MWh)", 50, 600, 300)
cost_per_kwh = st.sidebar.number_input("Cost per kWh (INR)", 0.1, 20.0, 8.0)
num_houses = st.sidebar.number_input("Number of Houses Represented", 1000, 10000000, 1000000)

source_type = st.sidebar.selectbox("Source Type", source_type_encoder.classes_)
region = st.sidebar.selectbox("Region", region_encoder.classes_)

# Filter data by date and labels
df_filtered = df[(df['datetime_iso'].dt.date >= start_date) &
                 (df['datetime_iso'].dt.date <= end_date) &
                 (df['source_type'] == source_type) & 
                 (df['region'] == region)]

# Encode categorical
df_filtered['source_type_encoded'] = source_type_encoder.transform(df_filtered['source_type'])
df_filtered['region_encoded'] = region_encoder.transform(df_filtered['region'])

# Fetch live weather data and override today's values if city matches
weather = fetch_weather(city)
st.subheader("Real-Time Weather Conditions")
# ✅ Predict demand using current weather

c1, c2, c3 = st.columns(3)
c1.metric("Temperature", f"{weather['temperature_c']} °C")
c2.metric("Humidity", f"{weather['humidity_percent']} %")
c3.metric("Wind Speed", f"{weather['wind_speed_kmph']} m/s")
if weather is None:
    st.error("Weather data not found for the input city.")
    st.stop()

today_idx = df_filtered[df_filtered['datetime_iso'].dt.date == pd.Timestamp.now().date()].index
if len(today_idx) > 0:
    idx = today_idx[0]
    df_filtered.at[idx, 'temperature_c'] = weather['temperature_c']
    df_filtered.at[idx, 'humidity_percent'] = weather['humidity_percent']
    df_filtered.at[idx, 'wind_speed_kmph'] = weather['wind_speed_kmph']
    df_filtered.at[idx, 'rainfall_mm'] = weather['rainfall_mm']

predictions = predict_demand(df_filtered)
df_filtered['predicted_demand'] = predictions
# ✅ Show current predicted electricity demand
try:
    current_predicted = df_filtered['predicted_demand'].iloc[-1]
    st.subheader("Current Predicted Electricity Demand")
    st.metric("Predicted Demand (MW)", f"{current_predicted:.2f}")
except:
    st.warning("Prediction unavailable — check input data.")


df_filtered['optimized_usage'] = optimize_grid(df_filtered['predicted_demand'], storage_capacity)
df_filtered['energy_saved_kwh'] = (df_filtered['predicted_demand'] - df_filtered['optimized_usage']) * 1000  # MWh to kWh

# Calculate total cost savings across filtered data
total_saved_kwh = df_filtered['energy_saved_kwh'].sum()
savings_per_house = calculate_cost_savings(total_saved_kwh, num_houses, cost_per_kwh)

# Tabs for visualization and info
tab_forecast, tab_optimize, tab_actions, tab_heatmap = st.tabs(["Forecast", "Optimization", "Actions", "Visualization"])

with tab_forecast:
    st.header("Energy Demand Forecast")

    fig1 = px.line(df_filtered, x='datetime_iso',
                   y=['total_demand_mw', 'predicted_demand'],
                   labels={"value": "Energy Demand (MW)", "datetime_iso": "Date"},
                   title="Historical and Predicted Energy Demand")
    st.plotly_chart(fig1, use_container_width=True)

    st.metric("Storage Capacity (MWh)", storage_capacity)
    st.metric("Total Energy Saved (kWh)", f"{total_saved_kwh:,.2f}")
    st.metric("Estimated Monthly Cost Savings Per Household (INR)", f"₹{savings_per_house:,.2f}")

with tab_optimize:
    st.header("Grid Optimization and Cost Savings")

    fig2 = px.area(df_filtered, x='datetime_iso', y=['predicted_demand', 'optimized_usage'],
                   labels={"value": "Energy Usage (MW)", "datetime_iso": "Date"},
                   title="Predicted vs Optimized Energy Usage")
    st.plotly_chart(fig2, use_container_width=True)

        # RECOMMENDED ENERGY SOURCE
    st.subheader("Recommended Energy Source")

    def select_source(temp, humidity, wind):
        if temp > 28:
            return "Solar"
        elif wind > 4:
            return "Wind"
        elif humidity > 75:
            return "Hydropower"
        else:
            return "Thermal"

    try:
        source_used = select_source(
            weather['temperature_c'],
            weather['humidity_percent'],
            weather['wind_speed_kmph']
        )
        st.success(f"Recommended source → {source_used}")
    except:
        st.warning("Weather data unavailable — cannot recommend source.")

with tab_actions:
    st.header("Charge/Discharge Recommendations")

    def recommend_action(row):
        if row['predicted_demand'] < 0.5 * storage_capacity:
            return "Charge"
        elif row['predicted_demand'] > 0.9 * storage_capacity:
            return "Discharge"
        else:
            return "Hold"

    df_filtered['action'] = df_filtered.apply(recommend_action, axis=1)
    colors = {'Charge': 'green', 'Discharge': 'red', 'Hold': 'gray'}

    fig3 = px.scatter(df_filtered, x='datetime_iso', y='predicted_demand', color='action',
                      color_discrete_map=colors,
                      labels={"predicted_demand": "Predicted Demand (MW)", "datetime_iso": "Date"},
                      title="Charging / Discharging Actions Over Time")
    st.plotly_chart(fig3, use_container_width=True)
    st.subheader("Battery Recommendation")

    THRESHOLD_HIGH = 1300      # High demand → Discharge
    THRESHOLD_LOW = 900        # Low demand → Charge

    def battery_action(predicted):
        if predicted > THRESHOLD_HIGH:
            return "DISCHARGE — High demand"
        elif predicted < THRESHOLD_LOW:
            return "CHARGE — Low demand"
        else:
            return "HOLD — Balanced"

    # Use latest predicted value
    latest_pred = df_filtered['predicted_demand'].iloc[-1]
    action = battery_action(latest_pred)

    if "DISCHARGE" in action:
        st.error(action)
    elif "CHARGE" in action:
        st.success(action)
    else:
        st.info(action)

with tab_heatmap:
    st.header("Visualization: Predicted Demand vs Temperature")
    # Remove NaNs and create meaningful bins
    valid_data = df_filtered.dropna(subset=['predicted_demand', 'temperature_c'])
    bins = pd.interval_range(start=valid_data['temperature_c'].min(),
                             end=valid_data['temperature_c'].max(),
                             freq=2)
    valid_data['temp_bin'] = pd.cut(valid_data['temperature_c'], bins)
    valid_data['temp_bin'] = valid_data['temp_bin'].astype(str)
    
    heatmap_data = valid_data.pivot_table(index='date',
                                          columns='temp_bin',
                                          values='predicted_demand',
                                          aggfunc='mean').fillna(0)
    
    fig4 = px.imshow(heatmap_data.T,  # transpose to have temp bins on x-axis visually
                     labels={'x': 'Date', 'y': 'Temperature Range (°C)', 'color': 'Avg Predicted Demand'},
                     title="Heatmap of Predicted Demand by Date and Temperature")
    st.plotly_chart(fig4, use_container_width=True)
    import plotly.graph_objects as go

    st.subheader("Temperature Category Pie Chart")

    # Ensure temperature exists
    if "temperature_c" in df_filtered.columns:

    # Define temp categories
        def classify_temp(t):
            if t < 15:
                return "Cold (<15°C)"
            elif 15 <= t < 25:
                return "Mild (15–25°C)"
            elif 25 <= t < 35:
                return "Warm (25–35°C)"
            else:
                return "Hot (>35°C)"

        df_filtered["temp_category"] = df_filtered["temperature_c"].apply(classify_temp)

        # Count category percentage
        temp_counts = df_filtered["temp_category"].value_counts()

        # Plot Pie Chart
        pie_fig = go.Figure(
            go.Pie(
                labels=temp_counts.index,
                values=temp_counts.values,
                hole=0.3
            )
        )

        pie_fig.update_layout(
            title="Distribution of Temperature Categories"
        )

        st.plotly_chart(pie_fig, use_container_width=True)

    else:
        st.warning("⚠ No 'temperature' column found!")





# Sidebar footer date/time
st.sidebar.markdown("---")
st.sidebar.markdown(f"Current IST Time: {now_ist()}")
