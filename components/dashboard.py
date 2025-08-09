import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from utils.visualization import create_risk_gauge, create_environmental_radar
from datetime import datetime

def render_dashboard(current_data, risk_prediction):
    """Render the main dashboard interface"""
    
    # Current status row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🌡️ Temperature",
            value=f"{current_data['temperature']:.1f}°C",
            delta=f"{current_data['temperature'] - 25:.1f}°C from normal"
        )
    
    with col2:
        st.metric(
            label="💧 Humidity",
            value=f"{current_data['humidity']:.1f}%",
            delta=f"{current_data['humidity'] - 60:.1f}% from normal"
        )
    
    with col3:
        st.metric(
            label="💨 Wind Speed",
            value=f"{current_data['wind_speed']:.1f} km/h",
            delta=f"{current_data['wind_speed'] - 12:.1f} km/h from normal"
        )
    
    with col4:
        # Risk status with color coding
        risk_level = risk_prediction['risk_level']
        risk_prob = risk_prediction['probability']
        
        if risk_level == 'HIGH':
            st.error(f"🔥 HIGH RISK\n{risk_prob:.1%} probability")
        elif risk_level == 'MEDIUM':
            st.warning(f"⚠️ MEDIUM RISK\n{risk_prob:.1%} probability")
        else:
            st.success(f"✅ LOW RISK\n{risk_prob:.1%} probability")
    
    st.markdown("---")
    
    # Main dashboard content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Current Risk Assessment")
        
        # Risk gauge
        risk_gauge = create_risk_gauge(risk_prediction)
        st.plotly_chart(risk_gauge, use_container_width=True)
        
        # Risk details
        st.markdown("### Risk Analysis")
        st.write(f"**Risk Level:** {risk_prediction['risk_level']}")
        st.write(f"**Probability:** {risk_prediction['probability']:.1%}")
        st.write(f"**Confidence:** {risk_prediction['confidence']:.1%}")
        
        # Recommendation based on risk level
        if risk_prediction['risk_level'] == 'HIGH':
            st.error("**⚠️ IMMEDIATE ACTION REQUIRED**\n- Evacuate personnel from high-risk areas\n- Activate fire suppression systems\n- Contact emergency services")
        elif risk_prediction['risk_level'] == 'MEDIUM':
            st.warning("**🔍 INCREASED MONITORING**\n- Increase patrol frequency\n- Prepare fire suppression equipment\n- Monitor weather conditions closely")
        else:
            st.success("**✅ NORMAL OPERATIONS**\n- Continue regular monitoring\n- Maintain fire prevention measures")
    
    with col2:
        st.subheader("🌐 Environmental Conditions")
        
        # Environmental radar chart
        radar_chart = create_environmental_radar(current_data)
        st.plotly_chart(radar_chart, use_container_width=True)
        
        # Detailed sensor readings
        st.markdown("### 📊 Detailed Sensor Readings")
        
        sensor_data = [
            ("🌡️ Temperature", f"{current_data['temperature']:.1f}°C", "°C"),
            ("💧 Humidity", f"{current_data['humidity']:.1f}%", "%"),
            ("🏔️ Pressure", f"{current_data['barometric_pressure']:.1f} hPa", "hPa"),
            ("🌱 Soil Moisture", f"{current_data['soil_moisture']:.1f}%", "%"),
            ("💨 Smoke Level", f"{current_data['smoke_level']:.0f} ppm", "ppm"),
            ("☀️ Sunlight", f"{current_data['sunlight_intensity']:.0f} lux", "lux"),
            ("🌪️ Wind Speed", f"{current_data['wind_speed']:.1f} km/h", "km/h")
        ]
        
        for icon, value, unit in sensor_data:
            st.write(f"{icon} **{value}**")
    
    st.markdown("---")
    
    # Additional dashboard sections
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📈 Trend Indicators")
        
        # Simple trend analysis (mock data for demonstration)
        trends = {
            "Temperature": "↗️ Rising",
            "Humidity": "↘️ Falling", 
            "Wind Speed": "↗️ Increasing",
            "Smoke Level": "➡️ Stable"
        }
        
        for param, trend in trends.items():
            st.write(f"**{param}:** {trend}")
    
    with col2:
        st.subheader("⏰ System Status")
        
        # System health indicators
        status_items = [
            ("🔋 Power", "Normal", "success"),
            ("📡 Communications", "Online", "success"),
            ("🛰️ GPS Signal", "Strong", "success"),
            ("💾 Data Storage", "87% Full", "warning"),
            ("🔄 Last Update", f"{datetime.now().strftime('%H:%M:%S')}", "info")
        ]
        
        for item, status, status_type in status_items:
            if status_type == "success":
                st.success(f"{item}: {status}")
            elif status_type == "warning":
                st.warning(f"{item}: {status}")
            else:
                st.info(f"{item}: {status}")
    
    with col3:
        st.subheader("🎯 Quick Actions")
        
        # Action buttons
        if st.button("🚨 Trigger Emergency Alert", type="primary"):
            st.session_state.emergency_triggered = True
            st.success("Emergency alert triggered!")
        
        if st.button("📊 Generate Report"):
            st.info("Report generation initiated...")
        
        if st.button("🔄 Calibrate Sensors"):
            st.info("Sensor calibration started...")
        
        if st.button("💾 Export Data"):
            st.info("Data export prepared...")
    
    # Data export functionality
    if st.checkbox("Show Raw Data"):
        st.subheader("📋 Raw Sensor Data")
        
        # Format current data for display
        display_data = {
            "Parameter": ["Temperature", "Humidity", "Barometric Pressure", 
                         "Soil Moisture", "Smoke Level", "Sunlight Intensity", "Wind Speed"],
            "Value": [
                f"{current_data['temperature']:.2f}",
                f"{current_data['humidity']:.2f}",
                f"{current_data['barometric_pressure']:.2f}",
                f"{current_data['soil_moisture']:.2f}",
                f"{current_data['smoke_level']:.2f}",
                f"{current_data['sunlight_intensity']:.0f}",
                f"{current_data['wind_speed']:.2f}"
            ],
            "Unit": ["°C", "%", "hPa", "%", "ppm", "lux", "km/h"],
            "Timestamp": [current_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')] * 7
        }
        
        import pandas as pd
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True)

def create_mini_trend_chart(values, title):
    """Create a mini trend chart for dashboard metrics"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=values,
        mode='lines',
        line=dict(color='blue', width=2),
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        height=150,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    
    return fig
