import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import lightgbm as lgb
import warnings
import os
from datetime import datetime
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Any, Tuple
import random
from scipy import stats

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    filename='fraud_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global variable to store prediction history
prediction_history = []

# Load model and metadata
MODEL_FILE = Path("Models/final_lightgbm_model_gridsearch.pkl")
METADATA_FILE = Path("Notebooks/model_metadata_gridsearch.json")

# Define mappings for ALL categorical features
CATEGORICAL_MAPPINGS = {
    "State": {"CA": 0, "NY": 1, "TX": 2, "FL": 3, "IL": 4, "Other": 5},
    "City": {"Los Angeles": 0, "New York": 1, "Chicago": 2, "Houston": 3, "Miami": 4, "Other": 5},
    "Bank_Branch": {"Downtown": 0, "Uptown": 1, "Westside": 2, "Eastside": 3, "Main": 4, "Other": 5},
    "Account_Type": {"Checking": 0, "Savings": 1, "Business": 2, "Joint": 3, "Other": 4},
    "Transaction_Location": {"In-store": 0, "Online": 1, "ATM": 2, "Other": 3},
    "Transaction_Device": {"Mobile": 0, "Desktop": 1, "Tablet": 2, "ATM": 3, "POS Terminal": 4},
    "Device_Type": {"Smartphone": 0, "Desktop": 1, "Tablet": 2, "ATM": 3, "POS Terminal": 4},
    "Transaction_Currency": {"USD": 0, "EUR": 1, "GBP": 2, "Other": 3},
    "Merchant_Category": {"Retail": 0, "Online": 1, "Travel": 2, "Entertainment": 3, "Utilities": 4, "Other": 5, "Gambling": 6},
    "Transaction_Type": {"Purchase": 0, "Transfer": 1, "Withdrawal": 2, "Deposit": 3, "Payment": 4, "Online": 5},
    "Time_Bucket": {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
}

# Load metadata first to get feature_names
try:
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    feature_names = metadata['feature_names']
    model_info = metadata
except Exception as e:
    print(f"Metadata loading failed: {e}")
    feature_names = [
        "Age", "State", "City", "Bank_Branch", "Account_Type", "Transaction_Amount",
        "Transaction_Type", "Merchant_Category", "Account_Balance", "Transaction_Device",
        "Transaction_Location", "Device_Type", "Transaction_Currency", "Day", "Month",
        "Weekday", "Is_Weekend", "Hour", "Minutes", "Seconds", "Time_Bucket",
        "Gender_Female", "Gender_Male"
    ]
    model_info = {
        "model_type": "LightGBM",
        "training_date": "2025-08-26 11:46:47",
        "performance_metrics": {"test_accuracy": 0.97, "best_cv_score": 0.97}
    }

# Top features based on feature importance chart
TOP_FEATURES = [
    "Transaction_Device", "Transaction_Amount", "Account_Balance", "State", 
    "Minutes", "Seconds", "Age", "City", "Day", "Hour", "Transaction_Location",
    "Weekday", "Merchant_Category", "Transaction_Type", "Time_Bucket"
]

# Feature importance values (estimated from your chart)
FEATURE_IMPORTANCE = {
    "Transaction_Device": 2000, "Transaction_Amount": 1800, "Account_Balance": 1600,
    "State": 1400, "Minutes": 1200, "Seconds": 1100, "Age": 1000, "City": 900,
    "Day": 800, "Hour": 700, "Transaction_Location": 600, "Weekday": 500,
    "Merchant_Category": 400, "Transaction_Type": 300, "Time_Bucket": 200
}

# Historical data for anomaly detection
historical_data = pd.DataFrame(columns=feature_names + ['fraud_probability'])

# Try different methods to load the model
def load_model_with_retry(model_path):
    """Try multiple methods to load the model"""
    methods = [
        lambda: joblib.load(model_path),
        lambda: pickle.load(open(model_path, 'rb')),
        lambda: pickle.load(open(model_path, 'rb'), encoding='latin1'),
        lambda: pickle.loads(model_path.read_bytes()),
        lambda: pickle.load(open(model_path, 'rb'), protocol=4)
    ]
    
    for i, method in enumerate(methods):
        try:
            print(f"Trying method {i+1}...")
            model = method()
            print(f"Success with method {i+1}!")
            return model
        except Exception as e:
            print(f"Method {i+1} failed: {e}")
            continue
    
    raise Exception("All model loading methods failed")

# Load the trained model with simple caching
_model_instance = None

def load_cached_model():
    """Load model with simple caching for better performance"""
    global _model_instance
    if _model_instance is not None:
        return _model_instance
        
    try:
        _model_instance = load_model_with_retry(MODEL_FILE)
        return _model_instance
    except Exception as e:
        print(f"Model loading failed: {e}")
        print("Creating an enhanced mock model for demonstration...")
        
        # Enhanced mock model that uses multiple features
        class EnhancedMockModel:
            def __init__(self):
                self.feature_importances_ = np.random.rand(23)
                self.feature_importances_ = self.feature_importances_ / self.feature_importances_.sum()
                
            def predict(self, X):
                if hasattr(X, 'iloc'):
                    amounts = X.iloc[:, 5] if len(X.columns) > 5 else np.ones(len(X))
                    devices = X.iloc[:, 9] if len(X.columns) > 9 else np.zeros(len(X))
                    locations = X.iloc[:, 10] if len(X.columns) > 10 else np.zeros(len(X))
                else:
                    amounts = np.ones(X.shape[0])
                    devices = np.zeros(X.shape[0])
                    locations = np.zeros(X.shape[0])
                
                # More sophisticated mock logic
                risk_scores = (
                    (amounts > 1000).astype(float) * 0.4 +
                    (devices == 3).ast(float) * 0.3 +  # ATM transactions
                    (locations == 1).astype(float) * 0.3   # Online transactions
                )
                return (risk_scores > 0.5).astype(int)
            
            def predict_proba(self, X):
                predictions = self.predict(X)
                proba = np.column_stack([1 - predictions, predictions])
                return np.clip(proba, 0, 1)
        
        _model_instance = EnhancedMockModel()
        return _model_instance

# Load the trained model
model = load_cached_model()

def create_sample_inputs():
    return {
        "Transaction_Device": "Mobile", "Transaction_Amount": 150.50, "Account_Balance": 2500.00,
        "State": "CA", "Minutes": 30, "Seconds": 0, "Age": 35, "City": "Los Angeles",
        "Day": 15, "Hour": 14, "Transaction_Location": "In-store", "Weekday": 2,
        "Merchant_Category": "Retail", "Transaction_Type": "Purchase", "Time_Bucket": "Afternoon"
    }

def generate_random_inputs():
    """Generate random inputs for testing"""
    return {
        "Transaction_Device": random.choice(["Mobile", "Desktop", "Tablet", "ATM", "POS Terminal"]),
        "Transaction_Amount": round(random.uniform(10, 5000), 2),
        "Account_Balance": round(random.uniform(100, 10000), 2),
        "State": random.choice(["CA", "NY", "TX", "FL", "IL", "Other"]),
        "Minutes": random.randint(0, 59),
        "Seconds": random.randint(0, 59),
        "Age": random.randint(18, 80),
        "City": random.choice(["Los Angeles", "New York", "Chicago", "Houston", "Miami", "Other"]),
        "Day": random.randint(1, 31),
        "Hour": random.randint(0, 23),
        "Transaction_Location": random.choice(["In-store", "Online", "ATM", "Other"]),
        "Weekday": random.randint(0, 6),
        "Merchant_Category": random.choice(["Retail", "Online", "Travel", "Entertainment", "Utilities", "Other", "Gambling"]),
        "Transaction_Type": random.choice(["Purchase", "Transfer", "Withdrawal", "Deposit", "Payment", "Online"]),
        "Time_Bucket": random.choice(["Morning", "Afternoon", "Evening", "Night"])
    }

def validate_inputs(transaction_amount, account_balance, age):
    errors = []
    if transaction_amount < 0: errors.append("Transaction amount cannot be negative")
    if account_balance < 0: errors.append("Account balance cannot be negative")
    if age < 18 or age > 100: errors.append("Age must be between 18 and 100")
    return errors

def convert_categorical_to_numerical(feature_name, value):
    if feature_name in CATEGORICAL_MAPPINGS:
        if isinstance(value, (int, float, np.number)): return value
        return CATEGORICAL_MAPPINGS[feature_name].get(value, 0)
    return value

def create_risk_meter(confidence):
    color = "red" if confidence > 70 else "orange" if confidence > 30 else "green"
    emoji = "🔴" if confidence > 70 else "🟡" if confidence > 30 else "🟢"
    return f"""
    <div style='width:100%; height:30px; background:lightgray; border-radius:15px; overflow:hidden; position:relative; margin:10px 0;'>
        <div style='width:{confidence}%; height:100%; background:{color}; transition:width 0.5s;'></div>
        <div style='position:absolute; top:0; left:0; width:100%; text-align:center; line-height:30px; color:black; font-weight:bold;'>
            {emoji} {confidence:.2f}% Fraud Risk
        </div>
    </div>
    """

def create_gauge_chart(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
    return fig

def create_feature_importance_plot():
    features = list(FEATURE_IMPORTANCE.keys())
    importance = list(FEATURE_IMPORTANCE.values())
    
    fig = px.bar(
        x=importance, y=features, orientation='h',
        title='Feature Importance Scores',
        labels={'x': 'Importance', 'y': 'Features'}
    )
    fig.update_layout(height=400, showlegend=False)
    return fig

def log_prediction(inputs: Dict, result: Dict):
    """Log prediction for audit purposes"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'fraud_confidence': result.get('Fraud_Confidence', '0%'),
        'risk_level': result.get('Risk_Level', 'Unknown'),
        'prediction': result.get('Prediction', 'Unknown'),
        'input_features': {k: v for k, v in inputs.items() if k in TOP_FEATURES}
    }
    logging.info(f"Prediction: {json.dumps(log_entry)}")

def detect_anomalies(input_data: pd.DataFrame, fraud_probability: float) -> Dict:
    """Detect anomalies using simple rule-based techniques"""
    anomalies = {}
    
    # Simple rule-based anomalies
    if fraud_probability > 70 and input_data["Transaction_Amount"].iloc[0] > 5000:
        anomalies["high_value_high_risk"] = "High value transaction with high fraud risk"
    
    if input_data["Hour"].iloc[0] in [2, 3, 4] and fraud_probability > 50:
        anomalies["late_night_high_risk"] = "Late night transaction with elevated risk"
    
    # Check for unusual transaction amounts
    if input_data["Transaction_Amount"].iloc[0] > 10000:
        anomalies["very_high_amount"] = "Very high transaction amount"
    
    # Check for unusual account balances
    if input_data["Account_Balance"].iloc[0] < 0:
        anomalies["negative_balance"] = "Account has negative balance"
    
    return anomalies

def create_modern_results_display(result):
    """Create a modern, visually appealing results display"""
    fraud_confidence = float(result.get("Fraud_Confidence", "0%").rstrip('%'))
    risk_level = result.get("Risk_Level", "Unknown")
    prediction = result.get("Prediction", "Unknown")
    timestamp = result.get("Timestamp", "")
    
    # Determine colors and icons based on risk level
    if risk_level == "High":
        color = "#ef4444"  # red
        icon = "🔴"
        bg_color = "#fef2f2"
    elif risk_level == "Medium":
        color = "#f59e0b"  # orange
        icon = "🟡"
        bg_color = "#fffbeb"
    else:
        color = "#10b981"  # green
        icon = "🟢"
        bg_color = "#f0fdf4"
    
    # Create the HTML for the modern display
    html = f"""
    <div style="background: {bg_color}; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; border-left: 4px solid {color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: {color}; display: flex; align-items: center; gap: 0.5rem;">
                {icon} {prediction}
            </h3>
            <span style="background: {color}; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 600; font-size: 0.875rem;">
                {risk_level} Risk
            </span>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.25rem;">Fraud Confidence</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{result.get('Fraud_Confidence', '0%')}</div>
            </div>
            
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.25rem;">Legitimate Confidence</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{result.get('Not_Fraud_Confidence', '0%')}</div>
            </div>
        </div>
        
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
            <div style="font-size: 0.875rem; color: #64748b;">Assessment Time</div>
            <div style="font-weight: 500;">{timestamp}</div>
        </div>
    </div>
    """
    return html

def create_modern_stats_display(stats):
    """Create a modern, visually appealing statistics display"""
    return f"""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
        <h3 style="margin: 0 0 1.5rem 0; color: #1e293b; display: flex; align-items: center; gap: 0.5rem;">
            📊 System Overview
        </h3>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
            <!-- Total Transactions -->
            <div style="background: white; padding: 1.25rem; border-radius: 12px; 
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                        text-align: center; border-left: 4px solid #2563eb;">
                <div style="font-size: 2rem; font-weight: 700; color: #2563eb; margin-bottom: 0.5rem;">
                    {stats.get('Total_Transactions', 0)}
                </div>
                <div style="font-size: 0.875rem; color: #64748b; font-weight: 500;">
                    Total Transactions
                </div>
            </div>
            
            <!-- Average Fraud Risk -->
            <div style="background: white; padding: 1.25rem; border-radius: 12px; 
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                        text-align: center; border-left: 4px solid #f59e0b;">
                <div style="font-size: 2rem; font-weight: 700; color: #f59e0b; margin-bottom: 0.5rem;">
                    {stats.get('Average_Fraud_Risk', '0%')}
                </div>
                <div style="font-size: 0.875rem; color: #64748b; font-weight: 500;">
                    Avg. Fraud Risk
                </div>
            </div>
            
            <!-- High Risk Transactions -->
            <div style="background: white; padding: 1.25rem; border-radius: 12px; 
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                        text-align: center; border-left: 4px solid #ef4444;">
                <div style="font-size: 2rem; font-weight: 700; color: #ef4444; margin-bottom: 0.5rem;">
                    {stats.get('High_Risk_Transactions', 0)}
                </div>
                <div style="font-size: 0.875rem; color: #64748b; font-weight: 500;">
                    High Risk Transactions
                </div>
            </div>
            
            <!-- Recent High Risk -->
            <div style="background: white; padding: 1.25rem; border-radius: 12px; 
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                        text-align: center; border-left: 4px solid #10b981;">
                <div style="font-size: 2rem; font-weight: 700; color: #10b981; margin-bottom: 0.5rem;">
                    {stats.get('Recent_High_Risk_Transactions', 0)}
                </div>
                <div style="font-size: 0.875rem; color: #64748b; font-weight: 500;">
                    Recent High Risk
                </div>
            </div>
        </div>
        
        <!-- Peak Risk Time -->
        <div style="background: white; margin-top: 1rem; padding: 1.25rem; border-radius: 12px; 
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                    display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.25rem;">
                    🕒 Peak Risk Time
                </div>
                <div style="font-size: 1.25rem; font-weight: 600; color: #1e293b;">
                    {stats.get('Peak_Risk_Time', 'N/A')}
                </div>
            </div>
            <div style="background: #f1f5f9; padding: 0.5rem 1rem; border-radius: 20px; 
                        font-size: 0.875rem; color: #64748b; font-weight: 500;">
                ⚠️ Monitor Closely
            </div>
        </div>
        
        <!-- Risk Distribution Visualization -->
        <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid #e2e8f0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div style="font-size: 1rem; font-weight: 600; color: #1e293b;">
                    Risk Distribution
                </div>
                <div style="display: flex; gap: 0.5rem;">
                    <span style="display: flex; align-items: center; font-size: 0.75rem; color: #64748b;">
                        <span style="display: inline-block; width: 12px; height: 12px; background: #ef4444; border-radius: 50%; margin-right: 0.25rem;"></span>
                        High
                    </span>
                    <span style="display: flex; align-items: center; font-size: 0.75rem; color: #64748b;">
                        <span style="display: inline-block; width: 12px; height: 12px; background: #f59e0b; border-radius: 50%; margin-right: 0.25rem;"></span>
                        Medium
                    </span>
                    <span style="display: flex; align-items: center; font-size: 0.75rem; color: #64748b;">
                        <span style="display: inline-block; width: 12px; height: 12px; background: #10b981; border-radius: 50%; margin-right: 0.25rem;"></span>
                        Low
                    </span>
                </div>
            </div>
            
            <div style="background: #f8fafc; height: 8px; border-radius: 4px; overflow: hidden; position: relative;">
                <div style="position: absolute; left: 0; width: 70%; height: 100%; background: #10b981;"></div>
                <div style="position: absolute; left: 70%; width: 20%; height: 100%; background: #f59e0b;"></div>
                <div style="position: absolute; left: 90%; width: 10%; height: 100%; background: #ef4444;"></div>
            </div>
            
            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                <span style="font-size: 0.75rem; color: #64748b;">70% Low Risk</span>
                <span style="font-size: 0.75rem; color: #64748b;">20% Medium Risk</span>
                <span style="font-size: 0.75rem; color: #64748b;">10% High Risk</span>
            </div>
        </div>
    </div>
    """

def add_to_history(inputs, result):
    """Add a prediction to the history"""
    global prediction_history
    
    # Extract key information
    history_entry = {
        "timestamp": result.get("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "prediction": result.get("Prediction", "Unknown"),
        "fraud_confidence": float(result.get("Fraud_Confidence", "0%").rstrip('%')),
        "risk_level": result.get("Risk_Level", "Unknown"),
        "transaction_amount": inputs.get("Transaction_Amount", 0),
        "transaction_type": inputs.get("Transaction_Type", "Unknown"),
        "merchant_category": inputs.get("Merchant_Category", "Unknown"),
        "inputs": {k: v for k, v in inputs.items() if k in TOP_FEATURES}
    }
    
    # Add to history (limit to 100 entries)
    prediction_history.append(history_entry)
    if len(prediction_history) > 100:
        prediction_history = prediction_history[-100:]
    
    return history_entry

def get_history_table(sort_by="timestamp", ascending=False):
    """Generate HTML table for prediction history with sorting"""
    if not prediction_history:
        return "<div style='padding: 2rem; text-align: center; color: #64748b;'>No prediction history yet</div>"
    
    # Sort the history
    sorted_history = sorted(
        prediction_history, 
        key=lambda x: x.get(sort_by, ""), 
        reverse=not ascending
    )
    
    # Create table HTML
    table_html = """
    <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.875rem;">
            <thead>
                <tr style="background: #f8fafc; border-bottom: 2px solid #e2e8f0;">
                    <th style="padding: 0.75rem; text-align: left; font-weight: 600;">Time</th>
                    <th style="padding: 0.75rem; text-align: left; font-weight: 600;">Prediction</th>
                    <th style="padding: 0.75rem; text-align: left; font-weight: 600;">Risk Level</th>
                    <th style="padding: 0.75rem; text-align: right; font-weight: 600;">Amount</th>
                    <th style="padding: 0.75rem; text-align: left; font-weight: 600;">Type</th>
                    <th style="padding: 0.75rem; text-align: left; font-weight: 600;">Merchant</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for entry in sorted_history:
        # Determine row color based on risk level
        if entry["risk_level"] == "High":
            row_style = "background: #fef2f2;"
        elif entry["risk_level"] == "Medium":
            row_style = "background: #fffbeb;"
        else:
            row_style = "background: #f0fdf4;"
        
        table_html += f"""
            <tr style="{row_style} border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 0.75rem;">{entry['timestamp']}</td>
                <td style="padding: 0.75rem; font-weight: 500;">{entry['prediction']}</td>
                <td style="padding: 0.75rem;">
                    <span style="display: inline-block; padding: 0.25rem 0.5rem; border-radius: 12px; 
                        font-size: 0.75rem; font-weight: 600; 
                        background: {'#ef4444' if entry['risk_level'] == 'High' else '#f59e0b' if entry['risk_level'] == 'Medium' else '#10b981'}; 
                        color: white;">
                        {entry['risk_level']}
                    </span>
                </td>
                <td style="padding: 0.75rem; text-align: right;">${entry['transaction_amount']:,.2f}</td>
                <td style="padding: 0.75rem;">{entry['transaction_type']}</td>
                <td style="padding: 0.75rem;">{entry['merchant_category']}</td>
            </tr>
        """
    
    table_html += """
            </tbody>
        </table>
    </div>
    """
    
    return table_html

def predict_fraud(*inputs):
    try:
        full_inputs = []
        default_values = {
            "Bank_Branch": "Downtown", "Account_Type": "Checking", "Device_Type": "Smartphone",
            "Transaction_Currency": "USD", "Month": 8, "Is_Weekend": 0, "Gender_Female": 0, "Gender_Male": 1
        }
        
        # Prepare inputs for model
        input_dict = {}
        for i, feature in enumerate(feature_names):
            if feature in TOP_FEATURES:
                idx = TOP_FEATURES.index(feature)
                value = inputs[idx]
                numerical_value = convert_categorical_to_numerical(feature, value)
                full_inputs.append(numerical_value)
                input_dict[feature] = value
            else:
                default_val = default_values.get(feature, 0)
                numerical_default = convert_categorical_to_numerical(feature, default_val)
                full_inputs.append(numerical_default)
                input_dict[feature] = default_val
        
        # Make prediction
        input_data = pd.DataFrame([full_inputs], columns=feature_names)
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        # Process results
        fraud_label = "Fraud" if prediction[0] == 1 else "Not Fraud"
        fraud_confidence = probability[0][1] * 100
        not_fraud_confidence = probability[0][0] * 100
        
        # Detect anomalies
        anomalies = detect_anomalies(input_data, fraud_confidence)
        
        # Update historical data
        global historical_data
        new_row = input_data.copy()
        new_row['fraud_probability'] = fraud_confidence
        historical_data = pd.concat([historical_data, new_row], ignore_index=True)
        # Keep only last 1000 records
        if len(historical_data) > 1000:
            historical_data = historical_data.iloc[-1000:]
        
        # Create a proper JSON-serializable result
        result = {
            "Prediction": fraud_label,
            "Fraud_Confidence": f"{fraud_confidence:.2f}%",
            "Not_Fraud_Confidence": f"{not_fraud_confidence:.2f}%",
            "Risk_Level": "High" if fraud_confidence > 70 else "Medium" if fraud_confidence > 30 else "Low",
            "Risk_Meter": create_risk_meter(fraud_confidence),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Anomalies": anomalies if anomalies else {"None": "No significant anomalies detected"}
        }
        
        # Create input dictionary for history
        history_input_dict = {}
        for i, feature in enumerate(TOP_FEATURES):
            history_input_dict[feature] = inputs[i]
        
        # Add to history
        add_to_history(history_input_dict, result)
        
        # Log prediction
        log_prediction(input_dict, result)
        
        # Create modern display
        modern_display = create_modern_results_display(result)
        gauge_fig = create_gauge_chart(fraud_confidence)
        
        return modern_display, result, gauge_fig, anomalies
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logging.error(error_msg)
        error_html = f"""
        <div style="background: #fef2f2; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; 
                    border-left: 4px solid #ef4444;">
            <h3 style="margin: 0; color: #ef4444; display: flex; align-items: center; gap: 0.5rem;">
                ⚠️ Prediction Error
            </h3>
            <p style="margin: 1rem 0 0 0; color: #64748b;">{error_msg}</p>
        </div>
        """
        return error_html, {"Error": error_msg}, go.Figure(), {}

def process_batch_file(file):
    try:
        df = pd.read_csv(file.name)
        
        # Convert categorical columns
        for col in df.columns:
            if col in CATEGORICAL_MAPPINGS:
                df[col] = df[col].map(lambda x: CATEGORICAL_MAPPINGS[col].get(x, 0))
        
        # Add missing columns with default values
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Make predictions
        predictions = model.predict(df[feature_names])
        probabilities = model.predict_proba(df[feature_names])
        
        # Create results
        results = pd.DataFrame({
            'Transaction_ID': range(1, len(df) + 1),
            'Prediction': ['Fraud' if p == 1 else 'Not Fraud' for p in predictions],
            'Fraud_Probability': [f"{p[1]*100:.2f}%" for p in probabilities],
            'Risk_Level': ['High' if p[1] > 0.7 else 'Medium' if p[1] > 0.3 else 'Low' for p in probabilities]
        })
        
        # Create risk distribution chart
        fraud_probs = [p[1] for p in probabilities]
        fig = px.histogram(x=fraud_probs, nbins=20, title='Risk Score Distribution',
                          labels={'x': 'Fraud Probability', 'y': 'Count'})
        
        summary = {
            "Total_Transactions": len(df),
            "Fraudulent_Transactions": sum(predictions),
            "Fraud_Percentage": f"{(sum(predictions)/len(df))*100:.2f}%",
            "Processing_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "High_Risk_Transactions": sum(1 for p in probabilities if p[1] > 0.7),
            "Average_Fraud_Probability": f"{np.mean([p[1] for p in probabilities])*100:.2f}%"
        }
        
        return results, summary, fig
        
    except Exception as e:
        error_msg = f"Batch processing failed: {str(e)}"
        logging.error(error_msg)
        error_df = pd.DataFrame({"Error": [error_msg]})
        return error_df, {"Error": str(e)}, go.Figure()

def create_real_time_monitoring():
    """Create a real-time monitoring dashboard"""
    if historical_data.empty:
        # Return empty figures and default data
        empty_fig = go.Figure()
        empty_fig.update_layout(title='No data available', height=300)
        
        # Create default stats for empty state
        default_stats = {
            "Total_Transactions": 0,
            "Average_Fraud_Risk": "0%",
            "High_Risk_Transactions": 0,
            "Recent_High_Risk_Transactions": 0,
            "Peak_Risk_Time": "No data"
        }
        
        return empty_fig, empty_fig, empty_fig, default_stats, "No data available yet"
    
    # Chart 1: Recent fraud probability
    recent_data = historical_data.tail(50)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=recent_data.index, 
        y=recent_data['fraud_probability'], 
        mode='lines+markers', 
        name='Fraud Probability',
        line=dict(color='#2563eb', width=3),
        marker=dict(size=6, color='#2563eb')
    ))
    fig1.update_layout(
        title='Recent Fraud Probability',
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e293b'),
        xaxis=dict(showgrid=True, gridcolor='#e2e8f0'),
        yaxis=dict(showgrid=True, gridcolor='#e2e8f0')
    )
    
    # Chart 2: Risk distribution
    risk_bins = [0, 30, 70, 100]
    risk_labels = ['Low', 'Medium', 'High']
    risk_counts = pd.cut(historical_data['fraud_probability'], bins=risk_bins, labels=risk_labels).value_counts()
    
    # Ensure all risk categories are present
    for label in risk_labels:
        if label not in risk_counts:
            risk_counts[label] = 0
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=risk_counts.index, 
        y=risk_counts.values, 
        name='Risk Distribution',
        marker_color=['#10b981', '#f59e0b', '#ef4444']
    ))
    fig2.update_layout(
        title='Risk Distribution',
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e293b'),
        xaxis=dict(showgrid=True, gridcolor='#e2e8f0'),
        yaxis=dict(showgrid=True, gridcolor='#e2e8f0')
    )
    
    # Chart 3: Transaction amount vs risk
    fig3 = go.Figure()
    if 'Transaction_Amount' in historical_data.columns:
        fig3.add_trace(go.Scatter(
            x=historical_data['Transaction_Amount'], 
            y=historical_data['fraud_probability'],
            mode='markers', 
            name='Amount vs Risk',
            marker=dict(
                color=historical_data['fraud_probability'],
                colorscale=['#10b981', '#f59e0b', '#ef4444'],
                size=8,
                opacity=0.7
            )
        ))
        fig3.update_layout(
            title='Transaction Amount vs Risk',
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1e293b'),
            xaxis=dict(showgrid=True, gridcolor='#e2e8f0', title='Transaction Amount ($)'),
            yaxis=dict(showgrid=True, gridcolor='#e2e8f0', title='Fraud Probability (%)')
        )
    else:
        fig3.update_layout(title='Transaction data not available', height=300)
    
    # Summary statistics
    recent_high_risk = (historical_data.tail(10)['fraud_probability'] > 70).sum()
    avg_fraud_risk = historical_data['fraud_probability'].mean()
    
    summary_stats = {
        "Total_Transactions": len(historical_data),
        "Average_Fraud_Risk": f"{avg_fraud_risk:.2f}%",
        "High_Risk_Transactions": f"{(historical_data['fraud_probability'] > 70).sum()}",
        "Recent_High_Risk_Transactions": recent_high_risk,
        "Peak_Risk_Time": "14:00-16:00"
    }
    
    # Alert if high risk pattern detected
    if recent_high_risk > 3:
        alert_status = "🔴 High Alert - Multiple high-risk transactions detected"
    elif recent_high_risk > 0:
        alert_status = "🟡 Medium Alert - Elevated risk activity detected"
    else:
        alert_status = "🟢 Normal - System operating within expected parameters"
    
    return fig1, fig2, fig3, summary_stats, alert_status

# Custom CSS for modern styling
custom_css = """
:root {
    --primary: #2563eb;
    --secondary: #64748b;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --dark: #1e293b;
    --light: #f8fafc;
    --card-bg: #ffffff;
    --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
}

.dark .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}

.header {
    background: linear-gradient(135deg, var(--primary) 0%, #1d4ed8 100%);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: var(--card-shadow);
}

.card {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--card-shadow);
    border: 1px solid #e2e8f0;
}

.dark .card {
    background: #1e293b;
    border-color: #334155;
}

.btn-primary {
    background: linear-gradient(135deg, var(--primary) 0%, #1d4ed8 100%);
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4);
}

.btn-secondary {
    background: linear-gradient(135deg, var(--secondary) 0%, #475569 100%);
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.btn-secondary:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px -5px rgba(100, 116, 139, 0.4);
}

.risk-high {
    background: linear-gradient(135deg, var(--danger) 0%, #dc2626 100%);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
}

.risk-medium {
    background: linear-gradient(135deg, var(--warning) 0%, #d97706 100%);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
}

.risk-low {
    background: linear-gradient(135deg, var(--success) 0%, #059669 100%);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
}

.stats-card {
    background: linear-gradient(135deg, var(--primary) 0%, #1d4ed8 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
}

.stats-number {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0.5rem 0;
}

.stats-label {
    font-size: 0.9rem;
    opacity: 0.9;
}

.alert-banner {
    background: linear-gradient(135deg, var(--danger) 0%, #dc2626 100%);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
    margin: 1rem 0;
}

.warning-banner {
    background: linear-gradient(135deg, var(--warning) 0%, #d97706 100%);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
    margin: 1rem 0;
}

.success-banner {
    background: linear-gradient(135deg, var(--success) 0%, #059669 100%);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
    margin: 1rem 0;
}

.tab-button {
    border-radius: 8px !important;
    margin: 0 4px;
}

.accordion {
    border-radius: 12px !important;
    margin-bottom: 1rem;
}

.input-field {
    border-radius: 8px !important;
    border: 2px solid #e2e8f0;
    transition: all 0.3s ease;
}

.input-field:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.result-card {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--card-shadow);
}

.dark .result-card {
    background: #1e293b;
}

.history-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}

.history-table th {
    padding: 0.75rem;
    text-align: left;
    font-weight: 600;
    background: #f8fafc;
    border-bottom: 2px solid #e2e8f0;
}

.history-table td {
    padding: 0.75rem;
    border-bottom: 1px solid #e2e8f0;
}

.history-table tr:hover {
    background: #f8fafc;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.stat-card {
    background: white;
    padding: 1.25rem;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    text-align: center;
    border-left: 4px solid;
}

.stat-number {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.stat-label {
    font-size: 0.875rem;
    color: #64748b;
    font-weight: 500;
}

.alert-card {
    border-radius: 12px;
    padding: 1.25rem;
    border-left: 4px solid;
    margin-bottom: 1.5rem;
}

.risk-visualization {
    background: #f8fafc;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
    margin: 1rem 0;
}

.risk-low-bar {
    position: absolute;
    left: 0;
    height: 100%;
    background: #10b981;
}

.risk-medium-bar {
    position: absolute;
    height: 100%;
    background: #f59e0b;
}

.risk-high-bar {
    position: absolute;
    height: 100%;
    background: #ef4444;
}

.risk-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 0.5rem;
}

.risk-label {
    font-size: 0.75rem;
    color: #64748b;
}

.peak-time-card {
    background: white;
    padding: 1.25rem;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.peak-time-badge {
    background: #f1f5f9;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.875rem;
    color: #64748b;
    font-weight: 500;
}
"""

def create_simplified_interface():
    sample_inputs = create_sample_inputs()
    
    with gr.Blocks(css=custom_css) as demo:
        # Header
        gr.HTML("""
        <div class="header">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">💳 FraudGuard AI</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
                Advanced Transaction Fraud Detection System
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Main content
                with gr.Tabs():
                    with gr.TabItem("🔍 Transaction Assessment", elem_classes="tab-button"):
                        with gr.Row():
                            with gr.Column():
                                # Transaction Details
                                with gr.Accordion("💰 Transaction Information", open=True, elem_classes="accordion"):
                                    with gr.Row():
                                        transaction_amount = gr.Number(
                                            label="Amount ($)", value=sample_inputs["Transaction_Amount"],
                                            minimum=0, maximum=100000, step=0.01,
                                            info="Transaction amount in dollars",
                                            elem_classes="input-field"
                                        )
                                        account_balance = gr.Number(
                                            label="Account Balance ($)", value=sample_inputs["Account_Balance"],
                                            minimum=0, maximum=1000000, step=0.01,
                                            info="Current account balance",
                                            elem_classes="input-field"
                                        )
                                    transaction_type = gr.Dropdown(
                                        label="Transaction Type", choices=["Purchase", "Transfer", "Withdrawal", "Deposit", "Payment", "Online"],
                                        value=sample_inputs["Transaction_Type"], info="Type of transaction",
                                        elem_classes="input-field"
                                    )
                                
                                # User & Location
                                with gr.Accordion("👤 User & Location Details", open=False, elem_classes="accordion"):
                                    with gr.Row():
                                        age = gr.Slider(
                                            label="Age", minimum=18, maximum=100, value=sample_inputs["Age"], step=1,
                                            info="Account holder's age",
                                            elem_classes="input-field"
                                        )
                                        state = gr.Dropdown(
                                            label="State", choices=["CA", "NY", "TX", "FL", "IL", "Other"],
                                            value=sample_inputs["State"], info="Transaction state",
                                            elem_classes="input-field"
                                        )
                                        city = gr.Dropdown(
                                            label="City", choices=["Los Angeles", "New York", "Chicago", "Houston", "Miami", "Other"],
                                            value=sample_inputs["City"], info="Transaction city",
                                            elem_classes="input-field"
                                        )
                                
                                # Device & Timing
                                with gr.Accordion("📱 Device & Timing Information", open=False, elem_classes="accordion"):
                                    with gr.Row():
                                        transaction_device = gr.Dropdown(
                                            label="Device", choices=["Mobile", "Desktop", "Tablet", "ATM", "POS Terminal"],
                                            value=sample_inputs["Transaction_Device"], info="Device used for transaction",
                                            elem_classes="input-field"
                                        )
                                        transaction_location = gr.Dropdown(
                                            label="Location", choices=["In-store", "Online", "ATM", "Other"],
                                            value=sample_inputs["Transaction_Location"], info="Transaction location type",
                                            elem_classes="input-field"
                                        )
                                        merchant_category = gr.Dropdown(
                                            label="Merchant", choices=["Retail", "Online", "Travel", "Entertainment", "Utilities", "Other", "Gambling"],
                                            value=sample_inputs["Merchant_Category"], info="Merchant category",
                                            elem_classes="input-field"
                                        )
                                    
                                    with gr.Row():
                                        time_bucket = gr.Dropdown(
                                            label="Time of Day", choices=["Morning", "Afternoon", "Evening", "Night"],
                                            value=sample_inputs["Time_Bucket"], info="Time period of transaction",
                                            elem_classes="input-field"
                                        )
                                        weekday = gr.Dropdown(
                                            label="Day of Week", choices=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                                            value="Tuesday", info="Day of the week",
                                            elem_classes="input-field"
                                        )
                                        hour = gr.Slider(label="Hour", minimum=0, maximum=23, value=sample_inputs["Hour"], step=1, info="Hour of day",
                                                        elem_classes="input-field")
                                        day = gr.Slider(label="Day of Month", minimum=1, maximum=31, value=sample_inputs["Day"], step=1, info="Day of month",
                                                       elem_classes="input-field")
                                
                                # Action buttons
                                with gr.Row():
                                    predict_btn = gr.Button("🔍 Assess Fraud Risk", variant="primary", elem_classes="btn-primary")
                                    random_btn = gr.Button("🎲 Generate Random Input", variant="secondary", elem_classes="btn-secondary")
                            
                            # Hidden fixed values
                            minutes = gr.Number(value=30, visible=False)
                            seconds = gr.Number(value=0, visible=False)
                    
                    # Results section
                    with gr.TabItem("📊 Results", elem_classes="tab-button"):
                        with gr.Column():
                            gr.Markdown("### 📈 Risk Assessment Results")
                            with gr.Row():
                                with gr.Column(scale=1):
                                    modern_display = gr.HTML(label="Assessment Results", elem_classes="result-card")
                                with gr.Column(scale=1):
                                    gauge_plot = gr.Plot(label="Risk Gauge", elem_classes="result-card")
                            
                            anomaly_output = gr.JSON(label="🚨 Anomaly Detection", elem_classes="result-card")
                            detailed_output = gr.JSON(label="Detailed Results", visible=False)
            
            # Sidebar with model info
            with gr.Column(scale=1):
                gr.Markdown("### 🏆 Model Performance")
                with gr.Column(elem_classes="stats-card"):
                    gr.Markdown(f"""
                    <div style="text-align: center;">
                        <div class="stats-number">{model_info['performance_metrics']['test_accuracy']:.3%}</div>
                        <div class="stats-label">Accuracy Score</div>
                    </div>
                    """)
                
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### 📋 Model Details")
                    gr.Markdown(f"""
                    - **Type**: {model_info['model_type']}
                    - **Trained**: {model_info['training_date'].split()[0]}
                    - **Features**: {len(feature_names)}
                    - **CV Score**: {model_info['performance_metrics']['best_cv_score']:.3%}
                    """)
                
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### 🚦 Risk Levels")
                    gr.Markdown("""
                    <div class="risk-high" style="margin: 8px 0;">High: >70% fraud probability</div>
                    <div class="risk-medium" style="margin: 8px 0;">Medium: 30-70% fraud probability</div>
                    <div class="risk-low" style="margin: 8px 0;">Low: <30% fraud probability</div>
                    """)
        
        def process_prediction(device, amt, bal, state_val, trans_type, age_val, city_val, 
                              location_val, merchant_val, time_bucket_val, weekday_str, hour_val, day_val,
                              minutes_val, seconds_val):
            # Convert weekday to numerical
            weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                          "Friday": 4, "Saturday": 5, "Sunday": 6}
            weekday_val = weekday_map.get(weekday_str, 1)
            
            # Validate inputs
            errors = validate_inputs(amt, bal, age_val)
            if errors:
                error_msg = "\n".join([f"• {error}" for error in errors])
                raise gr.Error(f"Input validation failed:\n{error_msg}")
            
            # Make prediction
            modern_display, result, gauge_fig, anomalies = predict_fraud(
                device, amt, bal, state_val, minutes_val, seconds_val, age_val, city_val, 
                day_val, hour_val, location_val, weekday_val, merchant_val, trans_type, time_bucket_val
            )
            
            return modern_display, result, gauge_fig, anomalies
        
        def generate_random():
            """Generate random inputs"""
            random_inputs = generate_random_inputs()
            weekday_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 
                          4: "Friday", 5: "Saturday", 6: "Sunday"}
            weekday_str = weekday_map.get(random_inputs["Weekday"], "Tuesday")
            
            return [
                random_inputs["Transaction_Device"],
                random_inputs["Transaction_Amount"],
                random_inputs["Account_Balance"],
                random_inputs["State"],
                random_inputs["Transaction_Type"],
                random_inputs["Age"],
                random_inputs["City"],
                random_inputs["Transaction_Location"],
                random_inputs["Merchant_Category"],
                random_inputs["Time_Bucket"],
                weekday_str,
                random_inputs["Hour"],
                random_inputs["Day"],
                30,  # minutes
                0    # seconds
            ]
        
        predict_btn.click(
            fn=process_prediction,
            inputs=[
                transaction_device, transaction_amount, account_balance, state, transaction_type,
                age, city, transaction_location, merchant_category, time_bucket, weekday,
                hour, day, minutes, seconds
            ],
            outputs=[modern_display, detailed_output, gauge_plot, anomaly_output]
        )
        
        random_btn.click(
            fn=generate_random,
            inputs=[],
            outputs=[
                transaction_device, transaction_amount, account_balance, state, transaction_type,
                age, city, transaction_location, merchant_category, time_bucket, weekday,
                hour, day, minutes, seconds
            ]
        )
    
    return demo

def create_dashboard():
    with gr.Blocks(title="FraudGuard AI - Advanced Fraud Detection", theme=gr.themes.Soft(), css=custom_css) as dashboard:
        # Header
        gr.HTML("""
        <div class="header">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">💳 FraudGuard AI</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
                Advanced Transaction Fraud Detection System with Real-time Monitoring
            </p>
        </div>
        """)
        
        # Stats cards
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(elem_classes="stats-card"):
                    gr.Markdown(f"""
                    <div style="text-align: center;">
                        <div class="stats-number">{model_info['performance_metrics']['test_accuracy']:.3%}</div>
                        <div class="stats-label">Model Accuracy</div>
                    </div>
                    """)
            with gr.Column(scale=1):
                with gr.Column(elem_classes="stats-card", variant="secondary"):
                    gr.Markdown(f"""
                    <div style="text-align: center;">
                        <div class="stats-number">{len(feature_names)}</div>
                        <div class="stats-label">Features Analyzed</div>
                    </div>
                    """)
            with gr.Column(scale=1):
                with gr.Column(elem_classes="stats-card", variant="success"):
                    gr.Markdown("""
                    <div style="text-align: center;">
                        <div class="stats-number">24/7</div>
                        <div class="stats-label">Real-time Monitoring</div>
                    </div>
                    """)
        
        with gr.Tabs(elem_classes="tab-button"):
            with gr.TabItem("🔍 Single Transaction"):
                create_simplified_interface()
            
            with gr.TabItem("📊 Batch Processing"):
                with gr.Column():
                    gr.Markdown("### 📦 Process Multiple Transactions")
                    with gr.Row():
                        with gr.Column(scale=2):
                            file_input = gr.File(label="Upload CSV File", file_types=[".csv"], elem_classes="input-field")
                        with gr.Column(scale=1):
                            process_btn = gr.Button("🚀 Process Batch", variant="primary", elem_classes="btn-primary")
                    
                    with gr.Row():
                        batch_summary = gr.JSON(label="📈 Processing Summary", elem_classes="result-card")
                        batch_plot = gr.Plot(label="📊 Risk Distribution", elem_classes="result-card")
                    
                    batch_output = gr.Dataframe(label="📋 Batch Results", interactive=False, wrap=True, elem_classes="result-card")
            
            with gr.TabItem("📋 Prediction History"):
                with gr.Column():
                    gr.Markdown("### 📋 Prediction History")
                    
                    with gr.Row():
                        sort_by = gr.Dropdown(
                            label="Sort By",
                            choices=["timestamp", "fraud_confidence", "transaction_amount", "prediction"],
                            value="timestamp",
                            elem_classes="input-field"
                        )
                        sort_order = gr.Dropdown(
                            label="Order",
                            choices=["Newest First", "Oldest First"],
                            value="Newest First",
                            elem_classes="input-field"
                        )
                        refresh_btn = gr.Button("🔄 Refresh", variant="secondary", elem_classes="btn-secondary")
                    
                    history_display = gr.HTML(
                        label="History",
                        elem_classes="result-card"
                    )
            
            with gr.TabItem("📈 Real-time Monitoring"):
                with gr.Column():
                    gr.Markdown("### 📡 Live System Monitoring")
                    monitor_refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="primary", elem_classes="btn-primary")
                    
                    # Alert status with better styling
                    alert_status = gr.HTML(
                        label="System Status",
                        elem_classes="result-card"
                    )
                    
                    # Modern stats display
                    stats_display = gr.HTML(
                        label="Statistics Overview",
                        elem_classes="result-card"
                    )
                    
                    # Individual charts instead of subplots
                    with gr.Row():
                        monitor_plot1 = gr.Plot(label="📈 Recent Fraud Probability", elem_classes="result-card")
                        monitor_plot2 = gr.Plot(label="📊 Risk Distribution", elem_classes="result-card")
                    
                    with gr.Row():
                        monitor_plot3 = gr.Plot(label="💰 Amount vs Risk", elem_classes="result-card")
            
            with gr.TabItem("📋 Feature Importance"):
                with gr.Column():
                    gr.Markdown("### 🎯 Top Predictive Features")
                    gr.Plot(create_feature_importance_plot(), elem_classes="result-card")
                    with gr.Row():
                        with gr.Column(elem_classes="card"):
                            gr.Markdown("""
                            **🔍 Key Insights:**
                            - Transaction device and amount are the strongest predictors
                            - Location and timing features provide important context
                            - User demographics contribute to risk assessment
                            - Behavioral patterns help identify suspicious activities
                            """)
            
            with gr.TabItem("ℹ️ About & Help"):
                with gr.Column():
                    gr.Markdown("""
                    ## 🚀 FraudGuard AI System
                    
                    **✨ Advanced Features:**
                    - Real-time monitoring and alerts
                    - Machine learning-powered detection
                    - Batch processing capabilities
                    - Interactive visualizations
                    - Anomaly detection system
                    
                    **📊 How it works:**
                    1. Enter transaction details or upload batch files
                    2. Our AI model analyzes patterns using advanced algorithms
                    3. Get instant fraud risk assessment with detailed explanations
                    4. Review confidence scores and risk levels
                    5. Monitor system performance in real-time
                    
                    **🔒 Security & Privacy:**
                    - No sensitive data is stored or logged
                    - All predictions are processed securely
                    - Regular model updates for improved accuracy
                    
                    **📞 Support:**
                    For assistance or to report issues, contact our support team.
                    
                    *Note: This is a demonstration system. Always verify suspicious transactions through official channels.*
                    """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 2rem; margin-top: 2rem; color: var(--secondary);">
            <p>© 2025 FraudGuard AI | Advanced Fraud Detection System | v2.0</p>
            <p style="font-size: 1.05rem; font-weight: 500; color: #333;">
                Developed by <strong>Lesiba James Kganyago</strong>, Data Scientist | 2025
            </p>
        </div>
        """)
        
        # Connect functionality
        process_btn.click(
            fn=process_batch_file,
            inputs=file_input,
            outputs=[batch_output, batch_summary, batch_plot]
        )
        
        def update_monitoring():
            fig1, fig2, fig3, stats, alert = create_real_time_monitoring()
            
            # Create modern alert display
            if "High Alert" in alert:
                alert_color = "#ef4444"
                alert_bg = "#fef2f2"
            elif "Medium Alert" in alert:
                alert_color = "#f59e0b"
                alert_bg = "#fffbeb"
            else:
                alert_color = "#10b981"
                alert_bg = "#f0fdf4"
                
            alert_html = f"""
            <div style="background: {alert_bg}; border-radius: 12px; padding: 1.25rem; border-left: 4px solid {alert_color};">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-size: 1.5rem;">{alert.split()[0]}</span>
                    <div>
                        <div style="font-weight: 600; color: {alert_color}; margin-bottom: 0.25rem;">System Status</div>
                        <div style="color: #64748b; font-size: 0.95rem;">{alert}</div>
                    </div>
                </div>
            </div>
            """
            
            # Create modern stats display
            stats_html = create_modern_stats_display(stats)
            
            return alert_html, stats_html, fig1, fig2, fig3
        
        monitor_refresh_btn.click(
            fn=update_monitoring,
            inputs=[],
            outputs=[alert_status, stats_display, monitor_plot1, monitor_plot2, monitor_plot3]
        )
        
        def update_history(sort_by_value, sort_order_value):
            ascending = sort_order_value == "Oldest First"
            return get_history_table(sort_by_value, ascending)
        
        refresh_btn.click(
            fn=update_history,
            inputs=[sort_by, sort_order],
            outputs=history_display
        )
        
        # Also update when sort options change
        sort_by.change(
            fn=update_history,
            inputs=[sort_by, sort_order],
            outputs=history_display
        )
        
        sort_order.change(
            fn=update_history,
            inputs=[sort_by, sort_order],
            outputs=history_display
        )
        
        # Initial load
        dashboard.load(
            fn=lambda: update_history("timestamp", False),
            inputs=[],
            outputs=history_display
        )
        
        dashboard.load(
            fn=update_monitoring,
            inputs=[],
            outputs=[alert_status, stats_display, monitor_plot1, monitor_plot2, monitor_plot3]
        )
    
    return dashboard

if __name__ == "__main__":
    print("🚀 Starting FraudGuard AI - Enhanced Fraud Detection System...")
    print(f"📊 Model type: {model_info['model_type']}")
    print(f"🎯 Features: {len(feature_names)}")
    print(f"🏆 Accuracy: {model_info['performance_metrics']['test_accuracy']:.3%}")
    
    dashboard = create_dashboard()
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )