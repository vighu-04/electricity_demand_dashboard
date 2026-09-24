# ⚡ Electricity Demand Prediction Dashboard

An interactive machine learning dashboard that predicts electricity demand based on historical loads, calendar events, and weather conditions. 

Built with **XGBoost** for prediction, **SHAP** for explainable AI (XAI) visualizations, **SQLite** for prediction logging, and **Streamlit** for the frontend interface.

## Features
* Live scenario simulation (Temperature, Humidity, Hour, Lag Features).
* Explainable AI: Waterfall plots showing exactly how each feature influenced the prediction.
* Anomaly Detection: Automatically flags unusually high or low grid demand spikes.
* Persistent Database: All simulations are logged into a local SQLite database.

## How to Run Locally
1. Clone this repository.
2. Create a virtual environment and activate it.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt