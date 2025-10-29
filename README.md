⚡ Smart Power Grid Optimization & Forecasting Dashboard

A real-time intelligent energy management dashboard built using Streamlit.
It predicts electricity demand, recommends optimal energy sources, suggests battery storage actions, and visualizes grid behavior using interactive charts.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

✅ Features
Feature	Description
🔥 Real-time Weather Integration	Fetches temperature, humidity, wind
⚡ Demand Prediction	Predicts electricity usage
🔋 Battery Control	Recommends charge / discharge
🔌 Energy Source Recommendation	Solar / Wind / Hydro / Thermal
📊 Data Visualization	Line, bar, pie, heatmap
📅 7-Day Forecast View	Grid demand forecast
✅ Grid Optimization	Calculates energy savings
🧮 Data Import	Load and process Excel/CSV

🛠 Tech Stack

Python

Streamlit

Pandas

NumPy

Plotly

OpenWeatherMap API


🔄 Workflow

Data is loaded from Excel

Weather is fetched via API

Demand is predicted

Best energy source is recommended

Battery charge/discharge logic calculated

Visual charts displayed in dashboard



📡 Weather API

The app fetches live:

Temperature

Humidity

Wind speed

Using OpenWeather API

🔑 Add your API key in app.py

🔮 Electricity Demand Prediction

Uses rule-based logic based on:

Temperature

Humidity

Wind speed

You can replace it with ML models if needed.

🔌 Source Recommendation Logic
Condition	Recommended Source
Temperature > 28°C	Solar
Wind > 4 m/s	Wind
Humidity > 75%	Hydropower
Else	Thermal
🔋 Battery Storage Logic
Condition	Action
Demand > High threshold	Discharge
Demand < Low threshold	Charge
In between	Idle

Battery system helps reduce grid stress during high-load periods.

📊 Visualizations

Included charts:

✅ Line chart – demand trend
✅ Bar chart
✅ Pie chart – source distribution
✅ Heatmap
✅ 7-day forecast chart
✅ Current weather panels

⚡ Optimized Demand

⚙ The model uses predicted demand → compares with battery availability → computes optimized grid usage and energy savings.



🔧 Installation
1️⃣ Clone repository
git clone <repo-url>
cd smart-grid-dashboard

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run App
streamlit run app.py

📁 Dataset

File: PROJECT_DATASET.xlsx

Expected fields:

datetime

demand

source

weather (optional)

Works with 2020 sample dataset.

✅ requirements.txt
streamlit
pandas
numpy
requests
plotly
openpyxl

OpenPyXL
