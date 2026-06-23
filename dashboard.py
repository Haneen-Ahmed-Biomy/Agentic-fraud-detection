import streamlit as st
import requests
import pandas as pd
import time
import random
import plotly.graph_objects as go
import plotly.express as px

# Setup Page Configuration
st.set_page_config(page_title="Agentic Fraud Dashboard", layout="wide")

st.title("🛡️ Agentic Fraud Detection: Six-Zone Monitoring Dashboard")
st.markdown("---")

# API Configuration
API_URL = "http://127.0.0.1:8000/predict"

# Sidebar for controls
st.sidebar.header("Simulation Settings")
run_sim = st.sidebar.button("Start Live Simulation")
sim_speed = st.sidebar.slider("Simulation Speed (seconds)", 0.5, 5.0, 2.0)

# Initialize Session State for tracking history
if 'history' not in st.session_state:
    st.session_state.history = []

# ==========================================
# Layout: The Six Monitoring Zones
# ==========================================

# First Row: Zones 1, 2, 3
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Zone 1: Cluster Health")
    cluster_chart = st.empty()

with col2:
    st.subheader("Zone 2: Live Scoring Feed")
    score_feed = st.empty()

with col3:
    st.subheader("Zone 3: Action Dispatch")
    action_chart = st.empty()

# Second Row: Zones 4, 5, 6
col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("Zone 4: SHAP Reliability")
    reliability_chart = st.empty()

with col5:
    st.subheader("Zone 5: Outcome Tracking")
    outcome_table = st.empty()

with col6:
    st.subheader("Zone 6: System Alerts")
    alert_box = st.empty()

# ==========================================
# Live Simulation Loop
# ==========================================
if run_sim:
    while True:
        # 1. Simulate a transaction (Random features for testing)
        payload = {
            "features": [
                random.uniform(100, 10000), # S (Amount)
                random.uniform(-1, 3),      # Gap
                random.uniform(0, 1),       # Mu
                random.uniform(0, 5),       # Feature X
                random.uniform(0, 1)        # Feature Y
            ]
        }
        
        try:
            # 2. Query the FastAPI Live ReAct Loop
            response = requests.post(API_URL, json=payload)
            data = response.json()
            st.session_state.history.append(data)
            
            # Keep only last 20 records for visual clarity
            if len(st.session_state.history) > 20:
                st.session_state.history.pop(0)
            
            df = pd.DataFrame(st.session_state.history)

            # Update Zone 1: Cluster Health (Counts per cluster)
            cluster_counts = df['assigned_cluster'].value_counts().reset_index()
            fig1 = px.pie(cluster_counts, values='count', names='assigned_cluster', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            cluster_chart.plotly_chart(fig1, use_container_width=True, key=f"pie_{time.time()}")

            # Update Zone 2: Live Scoring Feed
            recent_score = data['intelligent_score']
            score_feed.metric(label="Latest Intelligent Score", value=f"{recent_score:.2f}", delta=f"{recent_score - 0.5:.2f}")

            # Update Zone 3: Action Dispatch
            action_counts = df['action'].value_counts()
            action_chart.bar_chart(action_counts)

            # Update Zone 4: SHAP Reliability (Cosine Similarity Gauge)
            reliability = data['signals']['SHAP_cosine']
            fig4 = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = reliability,
                title = {'text': "SHAP Cosine Similarity"},
                gauge = {'axis': {'range': [0, 1]}, 'bar': {'color': "darkblue"}}
            ))
            reliability_chart.plotly_chart(fig4, use_container_width=True, key=f"gauge_{time.time()}")

            # Update Zone 5: Outcome Tracking (Table of Reasoning)
            outcome_df = df[['action', 'reasoning']].tail(5)
            outcome_table.table(outcome_df)

            # Update Zone 6: System Alerts
            if data['action'] == "BLOCK_CARD":
                alert_box.error(f"🚨 CRITICAL: {data['reasoning']}")
            elif data['action'] == "HUMAN_REVIEW":
                alert_box.warning(f"⚠️ ATTENTION: {data['reasoning']}")
            else:
                alert_box.success("✅ System Stable: Monitoring Normal Transactions")

        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
            break
        
        time.sleep(sim_speed)