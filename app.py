import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
from models.fire_risk_model import FireRiskPredictor
from utils.data_generator import SensorDataGenerator
from utils.visualization import create_risk_gauge, create_fire_spread_map
from components.dashboard import render_dashboard
from components.prediction import render_prediction_section
from components.historical import render_historical_analysis

# Page configuration
st.set_page_config(
    page_title="Forest Fire Monitoring & Prediction System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = SensorDataGenerator()
if 'fire_predictor' not in st.session_state:
    st.session_state.fire_predictor = FireRiskPredictor()
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

def main():
    # Title and header
    st.title("Smart Forest Fire Monitoring & Prediction System")
    st.markdown("---")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ System Controls")
        
        # Data source selection
        data_source = st.radio(
            "Data Source:",
            ["Live Simulation", "Manual Input"],
            help="Choose between live sensor simulation or manual data entry"
        )
        
        st.markdown("---")
        
        # Manual input section
        if data_source == "Manual Input":
            st.subheader("📊 Manual Sensor Input")
            
            temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.1)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, 0.1)
            barometric_pressure = st.slider("Barometric Pressure (hPa)", 950.0, 1050.0, 1013.25, 0.1)
            soil_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 40.0, 0.1)
            smoke_level = st.slider("Smoke Level (ppm)", 0.0, 1000.0, 50.0, 1.0)
            sunlight_intensity = st.slider("Sunlight Intensity (lux)", 0.0, 100000.0, 50000.0, 100.0)
            wind_speed = st.slider("Wind Speed (km/h)", 0.0, 100.0, 10.0, 0.1)
            
            manual_data = {
                'timestamp': datetime.now(),
                'temperature': temperature,
                'humidity': humidity,
                'barometric_pressure': barometric_pressure,
                'soil_moisture': soil_moisture,
                'smoke_level': smoke_level,
                'sunlight_intensity': sunlight_intensity,
                'wind_speed': wind_speed
            }
        
        # Auto-refresh settings
        st.markdown("---")
        st.subheader("🔄 Auto-Refresh")
        auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 1, 30, 5)
        
        # System status
        st.markdown("---")
        st.subheader("📡 System Status")
        st.success("✅ All sensors online")
        st.success("✅ ML model loaded")
        st.success("✅ Data logging active")
    
    # Main content area
    if data_source == "Live Simulation":
        current_data = st.session_state.data_generator.generate_current_reading()
    else:
        current_data = manual_data
    
    # Get risk prediction
    risk_prediction = st.session_state.fire_predictor.predict_risk(current_data)
    
    # Update historical data
    if len(st.session_state.historical_data) == 0:
        st.session_state.historical_data = st.session_state.data_generator.generate_historical_data(hours=24)
    else:
        new_row = pd.DataFrame([current_data])
        st.session_state.historical_data = pd.concat([st.session_state.historical_data, new_row], ignore_index=True)
        if len(st.session_state.historical_data) > 1000:  # Keep only last 1000 records
            st.session_state.historical_data = st.session_state.historical_data.tail(1000)
    
    # Check for alerts
    if risk_prediction['risk_level'] == 'HIGH':
        alert_msg = f"⚠️ HIGH FIRE RISK DETECTED - Probability: {risk_prediction['probability']:.1%}"
        if alert_msg not in st.session_state.alerts:
            st.session_state.alerts.append({
                'timestamp': current_data['timestamp'],
                'message': alert_msg,
                'level': 'HIGH'
            })
    
    # Display alerts
    if st.session_state.alerts:
        st.error(st.session_state.alerts[-1]['message'])
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predictions", "📈 Historical Analysis", "🚨 Alert System"])
    
    with tab1:
        render_dashboard(current_data, risk_prediction)
    
    with tab2:
        render_prediction_section(current_data, st.session_state.fire_predictor)
    
    with tab3:
        render_historical_analysis(st.session_state.historical_data)
    
    with tab4:
        render_alert_system(st.session_state.alerts)
    
    # Auto-refresh logic
    if auto_refresh and data_source == "Live Simulation":
        time.sleep(refresh_interval)
        st.rerun()

def render_alert_system(alerts):
    """Render the alert system interface"""
    st.subheader("🚨 Alert Management System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Recent Alerts")
        if alerts:
            # Display recent alerts
            for i, alert in enumerate(reversed(alerts[-10:])):  # Show last 10 alerts
                if alert['level'] == 'HIGH':
                    st.error(f"**{alert['timestamp'].strftime('%H:%M:%S')}** - {alert['message']}")
                else:
                    st.warning(f"**{alert['timestamp'].strftime('%H:%M:%S')}** - {alert['message']}")
        else:
            st.info("No alerts in the system")
    
    with col2:
        st.markdown("### Alert Settings")
        
        # Alert thresholds
        temp_threshold = st.number_input("Temperature Alert (°C)", value=35.0, min_value=20.0, max_value=50.0)
        humidity_threshold = st.number_input("Low Humidity Alert (%)", value=30.0, min_value=10.0, max_value=50.0)
        smoke_threshold = st.number_input("Smoke Alert (ppm)", value=200.0, min_value=50.0, max_value=500.0)
        
        # Notification settings
        st.markdown("### Notifications")
        email_alerts = st.checkbox("Email Alerts", value=True)
        sms_alerts = st.checkbox("SMS Alerts", value=False)
        push_alerts = st.checkbox("Push Notifications", value=True)
        
        if st.button("Clear All Alerts"):
            st.session_state.alerts = []
            st.success("All alerts cleared!")
            st.rerun()
    
    # Alert statistics
    if alerts:
        st.markdown("---")
        st.subheader("📊 Alert Statistics")
        
        # Create alert timeline
        alert_df = pd.DataFrame(alerts)
        alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])
        alert_df['hour'] = alert_df['timestamp'].dt.hour
        
        # Hourly alert distribution
        hourly_alerts = alert_df.groupby('hour').size().reset_index(name='count')
        
        fig = px.bar(
            hourly_alerts,
            x='hour',
            y='count',
            title="Alert Distribution by Hour",
            labels={'hour': 'Hour of Day', 'count': 'Number of Alerts'},
            color='count',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
