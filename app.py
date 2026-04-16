import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model files safely
try:
    base_path = os.path.dirname(__file__)
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
    kmeans = joblib.load(os.path.join(base_path, 'kmeans_model.pkl'))
    fraud_threshold = joblib.load(os.path.join(base_path, 'fraud_threshold.pkl'))
except FileNotFoundError:
    st.error("Model files not found. Please ensure required .pkl files are present.")
    st.stop()

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to check for potential fraud.")

# Sidebar inputs
with st.sidebar:
    st.header("Transaction Details")
    time = st.number_input("Time (seconds since first transaction)", value=0.0, format="%.2f")
    amount = st.number_input("Amount", value=0.0, format="%.2f")

    v_features = {}
    for i in range(1, 29):
        v_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.6f")

# Prepare input data
input_data = {'Time': time}
input_data.update(v_features)
input_data['Amount'] = amount

input_df = pd.DataFrame([input_data])

# Ensure correct feature order
feature_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
input_df = input_df[feature_order]

# Scale Amount correctly
input_df[['Amount']] = scaler.transform(input_df[['Amount']])

# Prediction
if st.button("Predict Fraud Risk"):
    predicted_cluster = kmeans.predict(input_df)[0]

    cluster_distances = kmeans.transform(input_df)
    distance_to_cluster_center = cluster_distances[0, predicted_cluster]

    fraud_risk = 1 if distance_to_cluster_center > fraud_threshold else 0

    st.subheader("Prediction Results:")
    st.write(f"Predicted Cluster: {predicted_cluster}")
    st.write(f"Distance to Cluster Center: {distance_to_cluster_center:.2f}")

    if fraud_risk == 1:
        st.error("⚠️ HIGH FRAUD RISK DETECTED!")
    else:
        st.success("✅ LOW FRAUD RISK")

    st.write("---")
    st.write("Note: This model uses clustering to detect anomalies, not guaranteed fraud.")
