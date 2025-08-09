import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from utils.visualization import create_fire_spread_map, create_feature_importance_chart
from datetime import datetime, timedelta

def render_prediction_section(current_data, fire_predictor):
    """Render the fire spread prediction section"""
    
    st.subheader("🔮 Fire Spread Predictions")
    
    # Get fire spread predictions
    spread_predictions = fire_predictor.predict_fire_spread(current_data, hours=[1, 2, 3])
    
    # Prediction overview
    col1, col2, col3 = st.columns(3)
    
    for i, (time_key, prediction) in enumerate(spread_predictions.items()):
        col = [col1, col2, col3][i]
        
        with col:
            risk_level = prediction['risk_prediction']['risk_level']
            risk_prob = prediction['risk_prediction']['probability']
            spread_radius = prediction['spread_radius']
            affected_area = prediction['affected_area']
            
            # Color coding based on risk level
            if risk_level == 'HIGH':
                st.error(f"**{time_key.upper()} PREDICTION**")
            elif risk_level == 'MEDIUM':
                st.warning(f"**{time_key.upper()} PREDICTION**")
            else:
                st.success(f"**{time_key.upper()} PREDICTION**")
            
            st.metric("Risk Level", risk_level)
            st.metric("Probability", f"{risk_prob:.1%}")
            st.metric("Spread Radius", f"{spread_radius:.0f}m")
            st.metric("Affected Area", f"{affected_area:.1f} ha")
    
    st.markdown("---")
    
    # Fire spread map
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Fire Spread Visualization")
        
        # Create and display fire spread map
        spread_map = create_fire_spread_map(spread_predictions)
        st.plotly_chart(spread_map, use_container_width=True)
        
        # Map legend and explanation
        st.markdown("""
        **Map Legend:**
        - 🔴 **Red Circle**: 3-hour spread prediction
        - 🟠 **Orange Circle**: 2-hour spread prediction  
        - 🟡 **Yellow Circle**: 1-hour spread prediction
        - 🔥 **Fire Icon**: Current sensor location (hypothetical fire origin)
        """)
    
    with col2:
        st.subheader("📊 Prediction Details")
        
        # Detailed prediction table
        prediction_data = []
        for time_key, prediction in spread_predictions.items():
            prediction_data.append({
                "Time": time_key,
                "Risk Level": prediction['risk_prediction']['risk_level'],
                "Probability": f"{prediction['risk_prediction']['probability']:.1%}",
                "Radius (m)": f"{prediction['spread_radius']:.0f}",
                "Area (ha)": f"{prediction['affected_area']:.1f}",
                "Temperature": f"{prediction['environmental_data']['temperature']:.1f}°C",
                "Humidity": f"{prediction['environmental_data']['humidity']:.1f}%",
                "Wind Speed": f"{prediction['environmental_data']['wind_speed']:.1f} km/h"
            })
        
        import pandas as pd
        df = pd.DataFrame(prediction_data)
        st.dataframe(df, use_container_width=True)
        
        # Environmental changes chart
        st.markdown("### 🌤️ Environmental Projections")
        
        times = ['Current', '1h', '2h', '3h']
        temp_values = [current_data['temperature']] + [spread_predictions[f'{i}h']['environmental_data']['temperature'] for i in [1, 2, 3]]
        humidity_values = [current_data['humidity']] + [spread_predictions[f'{i}h']['environmental_data']['humidity'] for i in [1, 2, 3]]
        wind_values = [current_data['wind_speed']] + [spread_predictions[f'{i}h']['environmental_data']['wind_speed'] for i in [1, 2, 3]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=times, y=temp_values, name='Temperature (°C)', 
                               mode='lines+markers', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=times, y=humidity_values, name='Humidity (%)', 
                               mode='lines+markers', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=times, y=wind_values, name='Wind Speed (km/h)', 
                               mode='lines+markers', line=dict(color='green')))
        
        fig.update_layout(
            title="Environmental Parameter Projections",
            xaxis_title="Time",
            yaxis_title="Value",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Advanced prediction analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 Model Insights")
        
        # Feature importance
        feature_importance = fire_predictor.get_feature_importance()
        importance_chart = create_feature_importance_chart(feature_importance)
        st.plotly_chart(importance_chart, use_container_width=True)
        
        st.markdown("""
        **Feature Importance Explanation:**
        - Shows which sensors contribute most to fire risk prediction
        - Higher values indicate more important factors
        - Helps understand what drives fire risk in current conditions
        """)
    
    with col2:
        st.subheader("📈 Risk Progression")
        
        # Risk progression over time
        times = ['Current', '1h', '2h', '3h']
        risk_probabilities = [fire_predictor.predict_risk(current_data)['probability']] + \
                           [spread_predictions[f'{i}h']['risk_prediction']['probability'] for i in [1, 2, 3]]
        
        colors = ['green' if p < 0.3 else 'orange' if p < 0.7 else 'red' for p in risk_probabilities]
        
        fig = go.Figure(data=[
            go.Bar(x=times, y=[p*100 for p in risk_probabilities], 
                  marker_color=colors,
                  text=[f"{p:.1%}" for p in risk_probabilities],
                  textposition='auto')
        ])
        
        fig.update_layout(
            title="Fire Risk Progression",
            xaxis_title="Time Horizon",
            yaxis_title="Fire Risk Probability (%)",
            height=300,
            showlegend=False
        )
        
        fig.add_hline(y=30, line_dash="dash", line_color="orange", 
                     annotation_text="Medium Risk Threshold")
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk progression insights
        if max(risk_probabilities) > 0.7:
            st.error("⚠️ **Critical**: Risk escalates to HIGH level")
        elif max(risk_probabilities) > 0.5:
            st.warning("⚠️ **Caution**: Risk may increase over time")
        else:
            st.success("✅ **Stable**: Risk remains at manageable levels")
    
    # Scenario analysis
    st.markdown("---")
    st.subheader("🎯 Scenario Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌡️ Temperature Impact")
        temp_scenarios = [current_data['temperature'] + i*2 for i in range(-2, 3)]
        temp_risks = []
        
        for temp in temp_scenarios:
            scenario_data = current_data.copy()
            scenario_data['temperature'] = temp
            risk = fire_predictor.predict_risk(scenario_data)
            temp_risks.append(risk['probability'])
        
        fig = go.Figure(data=[
            go.Scatter(x=temp_scenarios, y=[r*100 for r in temp_risks],
                      mode='lines+markers', line=dict(color='red', width=3))
        ])
        fig.update_layout(
            title="Risk vs Temperature",
            xaxis_title="Temperature (°C)",
            yaxis_title="Risk (%)",
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💧 Humidity Impact")
        humidity_scenarios = [max(10, current_data['humidity'] - 20 + i*10) for i in range(5)]
        humidity_risks = []
        
        for humidity in humidity_scenarios:
            scenario_data = current_data.copy()
            scenario_data['humidity'] = humidity
            risk = fire_predictor.predict_risk(scenario_data)
            humidity_risks.append(risk['probability'])
        
        fig = go.Figure(data=[
            go.Scatter(x=humidity_scenarios, y=[r*100 for r in humidity_risks],
                      mode='lines+markers', line=dict(color='blue', width=3))
        ])
        fig.update_layout(
            title="Risk vs Humidity",
            xaxis_title="Humidity (%)",
            yaxis_title="Risk (%)",
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### 💨 Wind Impact")
        wind_scenarios = [max(0, current_data['wind_speed'] + i*5) for i in range(-2, 3)]
        wind_risks = []
        
        for wind in wind_scenarios:
            scenario_data = current_data.copy()
            scenario_data['wind_speed'] = wind
            risk = fire_predictor.predict_risk(scenario_data)
            wind_risks.append(risk['probability'])
        
        fig = go.Figure(data=[
            go.Scatter(x=wind_scenarios, y=[r*100 for r in wind_risks],
                      mode='lines+markers', line=dict(color='green', width=3))
        ])
        fig.update_layout(
            title="Risk vs Wind Speed",
            xaxis_title="Wind Speed (km/h)",
            yaxis_title="Risk (%)",
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction confidence and limitations
    st.markdown("---")
    st.subheader("ℹ️ Prediction Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Model Performance")
        st.write(f"**Model Accuracy:** {fire_predictor.accuracy:.1%}")
        st.write(f"**Prediction Confidence:** {fire_predictor.predict_risk(current_data)['confidence']:.1%}")
        st.write("**Model Type:** Random Forest Classifier")
        st.write("**Features Used:** 7 environmental sensors")
    
    with col2:
        st.markdown("### ⚠️ Important Notes")
        st.warning("""
        **Prediction Limitations:**
        - Predictions are probabilistic estimates
        - Actual fire spread depends on many factors
        - Weather conditions can change rapidly
        - Human activities may influence outcomes
        - Use as guidance, not absolute truth
        """)
