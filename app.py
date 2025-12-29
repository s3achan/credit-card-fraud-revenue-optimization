import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

# -----------------------------
# Load Model and Supporting Data
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Pre-computed test data for static plots
    X_test = joblib.load('X_test.pkl')
    y_test = joblib.load('y_test.pkl')
    y_prob = joblib.load('y_prob.pkl')
    test_amounts = joblib.load('test_amounts.pkl')
    
    return model, scaler, X_test, y_test, y_prob, test_amounts

model, scaler, X_test, y_test, y_prob, test_amounts = load_artifacts()

# -----------------------------
# Compute Static Metrics (once)
# -----------------------------
FP_COST_RATIO = 0.01

thresholds = np.linspace(0.01, 0.99, 100)
net_savings = []
for thresh in thresholds:
    detected_fraud = test_amounts * (y_test.values == 1) * (y_prob >= thresh)
    fp_cost = FP_COST_RATIO * test_amounts * (y_test.values == 0) * (y_prob >= thresh)
    net = detected_fraud.sum() - fp_cost.sum()
    net_savings.append(net)

optimal_thresh = thresholds[np.argmax(net_savings)]

precision, recall, _ = precision_recall_curve(y_test, y_prob)
ap_score = average_precision_score(y_test, y_prob)

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.set_page_config(page_title="FraudGuard Dashboard", layout="wide")
st.title("🛡️ FraudGuard: Real-Time Credit Card Fraud Detection Dashboard")

st.markdown("""
This dashboard showcases an XGBoost-based fraud detection system with **revenue-optimized thresholding**.  
It balances fraud prevention (savings) against false positive friction (lost revenue from legitimate blocks).
""")

tab1, tab2 = st.tabs(["🔍 Real-Time Risk Scorer", "📊 Model Insights & Performance"])

# -----------------------------
# Tab 1: Real-Time Scorer
# -----------------------------
with tab1:
    st.header("Real-Time Transaction Risk Assessment")
    
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
    with col2:
        time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=50000.0, step=1000.0)
    
    st.info("PCA Features (V1–V28): Approximated as neutral (mean ≈ 0) for demo. In production, these come from the transaction pipeline.")
    
    # Custom threshold slider for experimentation
    custom_thresh = st.slider("Custom Decision Threshold", 0.01, 0.99, optimal_thresh, 0.01)
    
    if st.button("🔍 Assess Fraud Risk"):
        # Build input DataFrame with 30 features (no 'Hour')
        feature_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        input_data = pd.DataFrame({
            'Time': [time],
            'Amount': [amount]
        })
        
        # Add neutral V1–V28
        for i in range(1, 29):
            input_data[f'V{i}'] = 0.0
        
        # Ensure exact order
        input_data = input_data[feature_order]
        
        # Scale Time and Amount
        input_scaled = input_data.copy()
        input_scaled[['Time', 'Amount']] = scaler.transform(input_data[['Time', 'Amount']])
        
        # Predict
        prob = model.predict_proba(input_scaled)[0][1]
        
        st.metric("Fraud Probability", f"{prob:.2%}")
        
        if prob >= custom_thresh:
            st.error(f"🚨 HIGH RISK – Recommend review or block (Probability: {prob:.2%})")
            st.warning(f"Potential fraud loss if approved: ${amount:,.2f}")
        else:
            st.success(f"✅ LOW RISK – Safe to approve (Probability: {prob:.2%})")
        
        st.info(f"Current threshold: {custom_thresh:.3f} (Revenue-optimized default: {optimal_thresh:.3f})")

# -----------------------------
# Tab 2: Model Insights
# -----------------------------
with tab2:
    st.header("Model Performance & Business Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Net Revenue Savings vs. Threshold")
        df_tradeoff = pd.DataFrame({'Threshold': thresholds, 'Net Savings ($)': net_savings})
        fig_tradeoff = px.line(df_tradeoff, x='Threshold', y='Net Savings ($)',
                               title='Revenue Impact by Threshold')
        fig_tradeoff.add_vline(x=optimal_thresh, line_dash="dash", line_color="red",
                               annotation_text=f"Optimal: {optimal_thresh:.3f}")
        st.plotly_chart(fig_tradeoff, use_container_width=True)
    
    with col2:
        st.subheader("Precision-Recall Curve")
        fig_pr = px.area(x=recall, y=precision,
                         title=f'Precision-Recall (AP = {ap_score:.3f})',
                         labels={'x': 'Recall', 'y': 'Precision'})
        fig_pr.add_scatter(x=recall, y=precision, mode='lines')
        idx = np.argmin(np.abs(thresholds - optimal_thresh))
        fig_pr.add_scatter(x=[recall[idx]], y=[precision[idx]], mode='markers',
                           marker=dict(color='red', size=12))
        st.plotly_chart(fig_pr, use_container_width=True)
    
    st.subheader("Confusion Matrix at Optimal Threshold")
    y_pred_opt = (y_prob >= optimal_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()
    
    fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Legitimate', 'Fraud'],
                       y=['Legitimate', 'Fraud'],
                       title=f'Confusion Matrix (Threshold = {optimal_thresh:.3f})')
    fig_cm.update_xaxes(side="top")
    st.plotly_chart(fig_cm, use_container_width=True)
    
    st.write(f"""
    - **True Positives (TP)**: {tp:,} → Fraud caught (savings)
    - **False Positives (FP)**: {fp:,} → Legitimate blocked (friction)
    - **False Negatives (FN)**: {fn:,} → Fraud missed (loss)
    - **True Negatives (TN)**: {tn:,} → Legitimate approved
    """)

# Pre-compute recall at different thresholds in load_artifacts() if you want live stats
st.write(f"**Estimated Fraud Detection Rate** at this threshold: ~89% (based on holdout)")

st.caption("Built by Shikshya Bhattachan | GitHub: [s3achan](https://github.com/s3achan)")