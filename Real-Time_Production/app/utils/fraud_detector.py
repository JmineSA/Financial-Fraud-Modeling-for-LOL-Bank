import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = os.path.join(BASE_DIR, '..', 'Models', 'final_lightgbm_model_gridsearch.pkl')
METADATA_FILE = os.path.join(BASE_DIR, '..', 'Notebooks', 'model_metadata_gridsearch.json')

# Page config
st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="🚨",
    layout="wide"
)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None

@st.cache_resource
def load_model():
    """Load the fraud detection model"""
    try:
        detector = joblib.load(MODEL_FILE)
        
        # Load metadata if available
        try:
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
            required_features = metadata.get('required_features', [])
        except:
            required_features = [
                'amount', 'transaction_type', 'user_age', 'device_type', 
                'location', 'time_of_day', 'previous_frauds'
            ]
        
        return detector, required_features
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, []

# Load model
model, required_features = load_model()

# Header
st.title("🚨 Real-Time Fraud Detection System")
st.markdown("---")

if model is None:
    st.error("Model could not be loaded. Please check the model file path.")
else:
    # Create input form
    with st.form("fraud_detection_form"):
        st.header("Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Numeric inputs
            if 'amount' in required_features:
                amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=10.0)
            if 'user_age' in required_features:
                user_age = st.number_input("User Age", min_value=18, max_value=100, value=35)
            if 'previous_frauds' in required_features:
                previous_frauds = st.number_input("Previous Fraud Count", min_value=0, value=0)
        
        with col2:
            # Categorical inputs
            if 'transaction_type' in required_features:
                transaction_type = st.selectbox(
                    "Transaction Type",
                    ["online", "in-store", "atm", "transfer"],
                    index=0
                )
            if 'device_type' in required_features:
                device_type = st.selectbox(
                    "Device Type",
                    ["mobile", "desktop", "tablet", "unknown"],
                    index=0
                )
            if 'location' in required_features:
                location = st.selectbox(
                    "Location",
                    ["US", "UK", "EU", "ASIA", "OTHER"],
                    index=0
                )
            if 'time_of_day' in required_features:
                time_of_day = st.selectbox(
                    "Time of Day",
                    ["00-06", "06-12", "12-18", "18-24"],
                    index=2
                )
        
        # Submit button
        submitted = st.form_submit_button("🔍 Check for Fraud")
    
    # Process prediction when form is submitted
    if submitted:
        try:
            # Prepare input data
            input_data = {}
            for feature in required_features:
                if feature in locals():
                    input_data[feature] = locals()[feature]
                else:
                    input_data[feature] = 0  # Default value
            
            # Convert to DataFrame
            df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model.predict(df)
            probability = model.predict_proba(df)
            
            fraud_prob = float(probability[0][1])
            is_fraud = bool(prediction[0])
            
            # Display results
            st.markdown("---")
            st.header("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fraud Probability", f"{fraud_prob:.2%}")
            
            with col2:
                if is_fraud:
                    st.error("🚨 FRAUD DETECTED")
                else:
                    st.success("✅ Legitimate Transaction")
            
            with col3:
                risk_level = "High" if fraud_prob > 0.7 else "Medium" if fraud_prob > 0.3 else "Low"
                st.metric("Risk Level", risk_level)
            
            # Progress bar for probability
            st.progress(fraud_prob)
            
            # Additional info
            with st.expander("Detailed Analysis"):
                st.write("**Input Features:**")
                st.json(input_data)
                st.write(f"**Prediction Confidence:** {max(probability[0]):.2%}")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ System Info")
        st.write(f"Model loaded: ✅")
        st.write(f"Features required: {len(required_features)}")
        st.write("---")
        
        st.header("📊 Quick Actions")
        if st.button("🔄 Test Example"):
            st.info("Use the form above with example values")
        
        if st.button("📋 View Features"):
            st.write("Required features:", required_features)

# Footer
st.markdown("---")
st.caption("Real-Time Fraud Detection System | Built with Streamlit & LightGBM")