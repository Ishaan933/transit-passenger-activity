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

# Stop Number Dropdown
stop_numbers = sorted(df["stop_number"].unique())
stop_number = st.sidebar.selectbox("Select Stop Number", stop_numbers)

# Route Number and Name
filtered_data = df[df["stop_number"] == stop_number]
route_numbers = sorted(filtered_data["route_number"].unique())
route_number = st.sidebar.selectbox("Select Route Number", route_numbers)

route_names = sorted(filtered_data[filtered_data["route_number"] == route_number]["route_name"].unique())
route_name = st.sidebar.selectbox("Select Route Name", route_names)

day_type = st.sidebar.selectbox("Select Day Type", ["Weekday", "Saturday", "Sunday"])
time_period = st.sidebar.selectbox(
    "Select Time Period", ["Morning", "Mid-Day", "PM Peak", "Evening", "Night"]
)

# Prediction Logic
def predict_passenger_activity():
    year = 2025 if "2025" in schedule_period_name else 2026
    month = {"Spring": 4, "Summer": 7, "Fall": 10, "Winter": 1}[schedule_period_name.split()[0]]

    input_data = pd.DataFrame({
        "year": [year],
        "month": [month],
        "schedule_period_name": [encodings["schedule_period_name"].get(schedule_period_name, -1)],
        "route_number": [encodings["route_number"].get(route_number, -1)],
        "route_name": [encodings["route_name"].get(route_name, -1)],
        "day_type": [encodings["day_type"].get(day_type, -1)],
        "time_period": [encodings["time_period"].get(time_period, -1)],
    })

    # Predictions
    boardings_prediction = rf_boardings.predict(input_data)[0]
    alightings_prediction = rf_alightings.predict(input_data)[0]

    # Historical Data
    historical_data = df[
        (df["route_number"] == route_number) &
        (df["route_name"] == route_name) &
        (df["day_type"] == day_type) &
        (df["time_period"] == time_period)
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
            "route_number": latest["route_number"],
            "route_name": latest["route_name"],
            "day_type": latest["day_type"],
            "time_period": latest["time_period"],
            "average_boardings": latest["average_boardings"],
            "average_alightings": latest["average_alightings"],
            "total_boardings": historical_boardings,
            "total_alightings": historical_alightings,
            "weekdays": weekdays,
            "start_date": start_date,
            "end_date": end_date
        }
    else:
        historical_info = None

    total_weekdays = 86
    total_boardings = boardings_prediction * total_weekdays
    total_alightings = alightings_prediction * total_weekdays

    return {
        "boardings_prediction": boardings_prediction,
        "alightings_prediction": alightings_prediction,
        "total_boardings": total_boardings,
        "total_alightings": total_alightings,
        "historical": historical_info,
    }

# Run Prediction
if st.sidebar.button("Predict"):
    result = predict_passenger_activity()

    # Create columns for layout
    col1, col2 = st.columns(2)

    # Prediction Results
    with col1:
        st.subheader("Prediction Results")
        st.write(pd.DataFrame({
            "Metric": [
                "Average Boardings Prediction",
                "Average Alightings Prediction",
                "Total Predicted Boardings",
                "Total Predicted Alightings"
            ],
            "Value": [
                f"{result['boardings_prediction']:.2f}",
                f"{result['alightings_prediction']:.2f}",
                f"{result['total_boardings']:.2f}",
                f"{result['total_alightings']:.2f}"
            ]
        }))

        st.markdown("### Calculation Breakdown for Predictions:")
        st.markdown(f"- **Schedule Period:** 01/01/2025 to 04/30/2025")
        st.markdown(f"- **Total Weekdays in Schedule Period:** 86")
        st.markdown(f"- **Total Predicted Boardings:** {result['boardings_prediction']:.2f} × 86")
        st.markdown(f"- **Total Predicted Alightings:** {result['alightings_prediction']:.2f} × 86")

    # Historical Data
    with col2:
        st.subheader("Latest Historical Data")
        if result["historical"]:
            historical = result["historical"]
            st.write(pd.DataFrame({
                "Metric": [
                    "Schedule Period",
                    "Route Number",
                    "Route Name",
                    "Day Type",
                    "Time Period",
                    "Average Boardings",
                    "Average Alightings",
                    "Total Historical Boardings",
                    "Total Historical Alightings"
                ],
                "Value": [
                    historical["schedule_period"],
                    historical["route_number"],
                    historical["route_name"],
                    historical["day_type"],
                    historical["time_period"],
                    f"{historical['average_boardings']:.2f}",
                    f"{historical['average_alightings']:.2f}",
                    f"{historical['total_boardings']:.2f}",
                    f"{historical['total_alightings']:.2f}"
                ]
            }))

            st.markdown("### Calculation Breakdown for Historical Data:")
            st.markdown(f"- **Schedule Period Start Date:** {historical['start_date']}")
            st.markdown(f"- **Schedule Period End Date:** {historical['end_date']}")
            st.markdown(f"- **Total Weekdays in Schedule Period:** {historical['weekdays']}")
            st.markdown(f"- **Total Historical Boardings:** {historical['average_boardings']:.2f} × {historical['weekdays']}")
            st.markdown(f"- **Total Historical Alightings:** {historical['average_alightings']:.2f} × {historical['weekdays']}")
        else:
            st.write("No historical data available.")
