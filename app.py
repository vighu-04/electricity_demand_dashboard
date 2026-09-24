import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime

st.set_page_config(page_title="Electricity Demand Predictor", layout="wide", page_icon="⚡")

# ---- Custom styling ----
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background-color: #1c1f26;
        border: 1px solid #2d3139;
        border-radius: 10px;
        padding: 15px;
    }
    div[data-testid="stMetricValue"] { color: #4ade80; font-size: 1.8rem; }
    h1 { background: linear-gradient(90deg, #4ade80, #22d3ee);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
""", unsafe_allow_html=True)

def init_db():
    conn = sqlite3.connect('predictions.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, hour INTEGER, dayofweek INTEGER, month INTEGER,
            temperature REAL, humidity REAL, demand_lag_1 REAL, demand_lag_24 REAL,
            demand_lag_168 REAL, rolling_mean_24 REAL, predicted_demand REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(hour, dayofweek, month, temperature, humidity,
                    demand_lag_1, demand_lag_24, demand_lag_168, rolling_mean_24, prediction):
    conn = sqlite3.connect('predictions.db')
    conn.execute('''
        INSERT INTO predictions
        (timestamp, hour, dayofweek, month, temperature, humidity,
         demand_lag_1, demand_lag_24, demand_lag_168, rolling_mean_24, predicted_demand)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), hour, dayofweek, month, temperature, humidity,
          demand_lag_1, demand_lag_24, demand_lag_168, rolling_mean_24, float(prediction)))
    conn.commit()
    conn.close()

init_db()

# ---- Cached Resource Loading ----
@st.cache_resource
def load_artifacts():
    xgb_model = joblib.load('xgb_model.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    anomaly_stats = joblib.load('anomaly_stats.pkl')
    background = joblib.load('background_sample.pkl')
    explainer = shap.Explainer(xgb_model.predict, background)
    return xgb_model, feature_cols, anomaly_stats, explainer

xgb_model, feature_cols, anomaly_stats, explainer = load_artifacts()
# ---------------------------------

st.title("⚡ Electricity Demand Prediction Dashboard")
st.caption("AI-powered demand forecasting with explainable predictions and anomaly detection")

st.sidebar.header("🔧 Simulate a Scenario")
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
dayofweek = st.sidebar.selectbox("Day of Week", options=list(range(7)),
                                   format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
month = st.sidebar.slider("Month", 1, 12, 6)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0)

st.sidebar.markdown("**Recent Demand History**")
demand_lag_1 = st.sidebar.number_input("1 hour ago", value=5000.0)
demand_lag_24 = st.sidebar.number_input("Same time yesterday", value=5000.0)
demand_lag_168 = st.sidebar.number_input("Same time last week", value=5000.0)
rolling_mean_24 = st.sidebar.number_input("Avg, last 24 hrs", value=5000.0)

st.sidebar.write("---")
predict_button = st.sidebar.button("🔮 Predict Demand", use_container_width=True)

if predict_button:
    try:
        hour_sin, hour_cos = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
        month_sin, month_cos = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
        dow_sin, dow_cos = np.sin(2*np.pi*dayofweek/7), np.cos(2*np.pi*dayofweek/7)
        year, dayofyear = 2024, month * 30

        input_dict = {
            'year': year, 'dayofyear': dayofyear, 'Temperature': temperature, 'Humidity': humidity,
            'hour_sin': hour_sin, 'hour_cos': hour_cos, 'month_sin': month_sin, 'month_cos': month_cos,
            'dow_sin': dow_sin, 'dow_cos': dow_cos, 'demand_lag_1': demand_lag_1,
            'demand_lag_24': demand_lag_24, 'demand_lag_168': demand_lag_168, 'rolling_mean_24': rolling_mean_24
        }
        input_df = pd.DataFrame([input_dict])[feature_cols]
        prediction = xgb_model.predict(input_df)[0]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(label="Predicted Electricity Demand", value=f"{prediction:.0f} MW")

            st.subheader("Anomaly Check")
            if prediction > (rolling_mean_24 + anomaly_stats['threshold']):
                st.error(f"⚠️ Unusually HIGH vs 24hr avg ({rolling_mean_24:.0f} MW) — resembles a demand spike.")
            elif prediction < (rolling_mean_24 - anomaly_stats['threshold']):
                st.warning(f"⚠️ Unusually LOW vs 24hr avg ({rolling_mean_24:.0f} MW).")
            else:
                st.success(f"✅ Within normal expected range.")

        with col2:
            st.subheader("Why did the model predict this?")
            shap_explanation = explainer(input_df)
            shap_values_single = shap_explanation.values

            fig = plt.figure(figsize=(9, 4.5))
            shap.plots.waterfall(shap_explanation[0], show=False)
            st.pyplot(fig)
            plt.close(fig)

        with st.expander("📊 View detailed feature contributions"):
            contrib_df = pd.DataFrame({
                'Feature': feature_cols,
                'Value': input_df.iloc[0].values,
                'SHAP Impact': shap_values_single[0]
            }).sort_values('SHAP Impact', key=abs, ascending=False)
            st.dataframe(contrib_df.style.format({'Value': '{:.2f}', 'SHAP Impact': '{:.2f}'}), use_container_width=True)

        log_prediction(hour, dayofweek, month, temperature, humidity,
                        demand_lag_1, demand_lag_24, demand_lag_168, rolling_mean_24, prediction)
                        
    except Exception as e:
        st.error(f"Something went wrong while generating the prediction: {e}")

st.write("---")
with st.expander("📜 View Prediction History"):
    conn = sqlite3.connect('predictions.db')
    history_df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT 20", conn)
    conn.close()
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
    else:
        st.write("No predictions logged yet.")