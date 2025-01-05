---

# Transit Passenger Activity Prediction App

This **Streamlit-based application** predicts passenger activity (boardings and alightings) for public transit systems based on user-selected parameters such as schedule period, stop number, route number, route name, day type, and time period. Additionally, the app provides historical passenger activity data for the selected inputs when available.

---

## Features

- **Passenger Activity Prediction**:
  - Predicts average and total boardings and alightings for a given schedule period, stop, route, day type, and time period.
  - Calculates total boardings and alightings based on weekdays in the selected schedule period.
  
- **Historical Data Retrieval**:
  - Retrieves the most recent historical data for the selected parameters and calculates total boardings and alightings for that historical period.

- **Time Period Filtering**:
  - Filters predictions and historical data by the selected time period (Morning, Mid-Day, PM Peak, Evening, Night).

- **Side-by-Side Display**:
  - Displays prediction results and historical data in two columns for easy comparison.

- **Detailed Calculations**:
  - Displays intermediate calculations for both predictions and historical data, such as weekdays in the schedule period and total activity calculations.

---

## Folder Structure

```
.
├── .devcontainer/                   # Optional: Configuration for VS Code remote container
├── .github/                         # Optional: GitHub-specific configurations
├── dataset/                         # Folder for dataset files
│   ├── stop_10637_data.csv
├── models/                          # Folder for model and encoding files
│   ├── encodings.pkl
│   ├── rf_boardings.pkl
│   ├── rf_alightings.pkl
├── .gitignore                       # Ignored files for Git
├── LICENSE                          # License for the project
├── README.md                        # Project documentation
├── requirements.txt                 # Python package dependencies
├── streamlit_app.py                 # Main Streamlit application
```

---

## Requirements

- Python 3.7 or higher
- Required Python packages:
  - `streamlit`
  - `pandas`
  - `joblib`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/transit-passenger-activity.git
   cd transit-passenger-activity
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following files are in the correct directories:
   - **Dataset** (`stop_10637_data.csv`) in the `dataset/` folder.
   - **Models and Encodings** (`encodings.pkl`, `rf_boardings.pkl`, `rf_alightings.pkl`) in the `models/` folder.

---

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Use the sidebar to:
   - Select a **Schedule Period** (e.g., Summer 2025, Winter 2026).
   - Choose a **Stop Number**, **Route Number**, and **Route Name**.
   - Select a **Day Type** (Weekday, Saturday, Sunday).
   - Choose a **Time Period** (Morning, Mid-Day, PM Peak, Evening, Night).

3. Click the **Predict** button to view:
   - **Prediction Results**: Displays average and total predicted boardings and alightings for the selected parameters.
   - **Latest Historical Data**: Displays historical boardings and alightings data, including calculations for the total historical activity.

---

## How It Works

1. **Prediction Results**:
   - Uses pre-trained Random Forest models (`rf_boardings.pkl`, `rf_alightings.pkl`) to predict average boardings and alightings.
   - Encodes categorical inputs (e.g., schedule period, route number) using mappings in `encodings.pkl`.

2. **Historical Data**:
   - Retrieves historical data for the selected stop, route, day type, and time period from the dataset (`stop_10637_data.csv`).
   - Calculates total historical boardings and alightings based on the number of weekdays in the historical schedule period.

3. **Dynamic Weekday Calculation**:
   - Calculates the number of weekdays dynamically for both prediction and historical periods.

---

## Example Screenshots

### Input Parameters
![Input Parameters](path/to/screenshot_input_parameters.png)

### Prediction and Historical Data
![Prediction and Historical Data](path/to/screenshot_prediction_and_historical_data.png)

---

## Notes

- Ensure the dataset file (`stop_10637_data.csv`) is accurate and up-to-date for reliable historical data.
- The `Time Period` filtering is essential for precise predictions and historical data retrieval.

---

## Future Improvements

- Add additional filters (e.g., weather, special events) to enhance prediction accuracy.
- Incorporate visualizations such as charts or graphs for better data presentation.
- Add support for real-time data streams for dynamic predictions.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---
