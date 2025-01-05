import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta

# Paths to files
DATASET_PATH = "dataset/stop_10637_data.csv"
RF_BOARDINGS_PATH = "models/rf_boardings.pkl"
RF_ALIGHTINGS_PATH = "models/rf_alightings.pkl"
ENCODINGS_PATH = "models/encodings.pkl"

# Load dataset, models, and encodings
@st.cache_data
def load_data():
    return pd.read_csv(DATASET_PATH)

@st.cache_resource
def load_models():
    rf_boardings = joblib.load(RF_BOARDINGS_PATH)
    rf_alightings = joblib.load(RF_ALIGHTINGS_PATH)
    encodings = joblib.load(ENCODINGS_PATH)
    return rf_boardings, rf_alightings, encodings

df = load_data()
rf_boardings, rf_alightings, encodings = load_models()

# Sidebar inputs
st.sidebar.header("Passenger Activity Prediction")
schedule_period_name = st.sidebar.selectbox(
    "Select Schedule Period",
    ["Summer 2025", "Fall 2025", "Spring 2025", "Winter 2026"]
)
stop_number = st.sidebar.number_input(
    "Enter Stop Number", value=10637, min_value=1, step=1
)

# Filter route numbers based on stop number
filtered_data = df[df["stop_number"] == stop_number]
route_numbers = sorted(filtered_data["route_number"].unique())
route_number = st.sidebar.selectbox("Select Route Number", route_numbers)

# Filter route names based on route number
route_names = sorted(filtered_data[filtered_data["route_number"] == route_number]["route_name"].unique())
route_name = st.sidebar.selectbox("Select Route Name", route_names)

day_type = st.sidebar.selectbox("Select Day Type", ["Weekday", "Saturday", "Sunday"])
time_period = st.sidebar.selectbox(
    "Select Time Period", ["Morning", "Mid-Day", "PM Peak", "Evening", "Night"]
)

# Prediction logic
def predict_passenger_activity():
    # Extract year and month
    year = 2025 if "2025" in schedule_period_name else 2026
    month = {"Spring": 4, "Summer": 7, "Fall": 10, "Winter": 1}[schedule_period_name.split()[0]]

    # Handle unseen schedule periods
    if schedule_period_name not in encodings["schedule_period_name"]:
        max_encoding = max(encodings["schedule_period_name"].values())
        encodings["schedule_period_name"][schedule_period_name] = max_encoding + 1

    input_data = pd.DataFrame({
        "year": [year],
        "month": [month],
        "schedule_period_name": [encodings["schedule_period_name"].get(schedule_period_name, -1)],
        "route_number": [encodings["route_number"].get(route_number, -1)],
        "route_name": [encodings["route_name"].get(route_name, -1)],
        "day_type": [encodings["day_type"].get(day_type, -1)],
        "time_period": [encodings["time_period"].get(time_period, -1)],
    })

    # Predict boardings and alightings
    boardings_prediction = rf_boardings.predict(input_data)[0]
    alightings_prediction = rf_alightings.predict(input_data)[0]

    # Historical data analysis
    historical_data = df[
        (df["route_number"] == route_number)
        & (df["route_name"] == route_name)
        & (df["day_type"] == day_type)
        & (df["time_period"] == time_period)
    ].sort_values("schedule_period_start_date", ascending=False)

    if not historical_data.empty:
        latest = historical_data.iloc[0]
        start_date = datetime.strptime(latest["schedule_period_start_date"], "%m/%d/%Y %I:%M:%S %p")
        end_date = datetime.strptime(latest["schedule_period_end_date"], "%m/%d/%Y %I:%M:%S %p")

        weekdays = sum(1 for d in pd.date_range(start_date, end_date) if d.weekday() < 5)
        historical_boardings = latest["average_boardings"] * weekdays
        historical_alightings = latest["average_alightings"] * weekdays

        historical_info = {
            "schedule_period": latest["schedule_period_name"],
            "boardings": historical_boardings,
            "alightings": historical_alightings,
            "weekdays": weekdays,
        }
    else:
        historical_info = None

    # Predict total boardings/alightings for the current schedule period
    weekdays = sum(
        1
        for d in pd.date_range(datetime(2025, 1, 1), datetime(2025, 4, 30))
        if d.weekday() < 5
    )
    total_boardings = boardings_prediction * weekdays
    total_alightings = alightings_prediction * weekdays

    return {
        "boardings_prediction": boardings_prediction,
        "alightings_prediction": alightings_prediction,
        "total_boardings": total_boardings,
        "total_alightings": total_alightings,
        "historical": historical_info,
    }

# Run prediction
if st.sidebar.button("Predict"):
    result = predict_passenger_activity()

    st.subheader("Prediction Results")
    st.write(f"**Average Boardings Prediction:** {result['boardings_prediction']:.2f}")
    st.write(f"**Average Alightings Prediction:** {result['alightings_prediction']:.2f}")
    st.write(f"**Total Predicted Boardings:** {result['total_boardings']:.2f}")
    st.write(f"**Total Predicted Alightings:** {result['total_alightings']:.2f}")

    # Display historical data
    if result["historical"]:
        st.subheader("Historical Data")
        st.write(result["historical"])
    else:
        st.write("No historical data available.")
