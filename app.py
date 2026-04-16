import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained scaler, KMeans model, and fraud threshold
try:
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
    fraud_threshold = joblib.load('fraud_threshold.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'scaler.pkl', 'kmeans_model.pkl', and 'fraud_threshold.pkl' are in the same directory.")
    st.stop()

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to check for potential fraud.")

# Create input fields for all features (V1-V28, Time, Amount)
with st.sidebar:
    st.header("Transaction Details")
    time = st.number_input("Time (seconds since first transaction)", value=0.0, format="%.2f")
    amount = st.number_input("Amount", value=0.0, format="%.2f")

    # Generate V features dynamically
    v_features = {}
    for i in range(1, 29):
        v_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.6f")

# Combine inputs into a dictionary
input_data = {'Time': time}
input_data.update(v_features)
input_data['Amount'] = amount

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess the 'Amount' feature using the loaded scaler
input_df['Amount'] = scaler.transform(input_df[['Amount']])

if st.button("Predict Fraud Risk"):
    # Predict the cluster for the input transaction
    predicted_cluster = kmeans.predict(input_df)[0]

    # Calculate the distance to the center of the predicted cluster
    cluster_distances = kmeans.transform(input_df)
    distance_to_cluster_center = cluster_distances[0, predicted_cluster]

    # Determine fraud risk based on the threshold
    fraud_risk = 1 if distance_to_cluster_center > fraud_threshold else 0

    st.subheader("Prediction Results:")
    st.write(f"**Predicted Cluster:** {predicted_cluster}")
    st.write(f"**Distance to Cluster Center:** {distance_to_cluster_center:.2f}")

    if fraud_risk == 1:
        st.error("\u26a0\ufe0f **HIGH FRAUD RISK DETECTED!** This transaction is significantly far from typical cluster behavior.")
    else:
        st.success("\u2705 **LOW FRAUD RISK.** This transaction appears to be within normal cluster boundaries.")

    st.write("---")
    st.write("**Note:** This model uses unsupervised clustering to identify anomalies. High risk indicates unusual patterns, not definitive fraud.")
