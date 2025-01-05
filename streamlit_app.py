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

# App title
st.title("Passenger Activity Predictions")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
schedule_period_name = st.sidebar.selectbox(
    "Select Schedule Period",
    ["Summer 2025", "Fall 2025", "Spring 2025", "Winter 2026"]
)
route_number = st.sidebar.selectbox(
    "Select Route Number",
    sorted(df["route_number"].unique())
)
route_name = st.sidebar.selectbox(
    "Select Route Name",
    sorted(df[df["route_number"] == route_number]["route_name"].unique())
)
day_type = st.sidebar.selectbox("Select Day Type", ["Weekday", "Saturday", "Sunday"])
time_period = st.sidebar.selectbox(
    "Select Time Period", ["Morning", "Mid-Day", "PM Peak", "Evening", "Night"]
)

# Prediction logic
def predict_passenger_activity():
    # Extract year and month
    year = 2025 if "2025" in schedule_period_name else 2026
    month = {"Spring": 4, "Summer": 7, "Fall": 10, "Winter": 1}[schedule_period_name.split()[0]]

    # Prepare input data
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

    return boardings_prediction, alightings_prediction

# Display results
if st.sidebar.button("Predict"):
    boardings_prediction, alightings_prediction = predict_passenger_activity()

    # Display predictions
    st.subheader("Prediction Results")
    st.write(f"**Average Boardings Prediction:** {boardings_prediction:.2f}")
    st.write(f"**Average Alightings Prediction:** {alightings_prediction:.2f}")

    # Additional breakdown
    st.write("**Calculation Breakdown:**")
    st.write(f"- Schedule Period: {schedule_period_name}")
    st.write(f"- Route Number: {route_number}")
    st.write(f"- Route Name: {route_name}")
    st.write(f"- Day Type: {day_type}")
    st.write(f"- Time Period: {time_period}")

