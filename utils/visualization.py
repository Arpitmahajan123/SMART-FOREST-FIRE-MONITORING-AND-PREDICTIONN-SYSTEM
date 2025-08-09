import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def create_risk_gauge(risk_prediction):
    """Create a risk level gauge chart"""
    risk_value = risk_prediction['probability'] * 100
    color = risk_prediction['color']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fire Risk Level (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "lightgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"},
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_fire_spread_map(spread_predictions):
    """Create fire spread visualization map"""
    fig = go.Figure()
    
    # Center point (hypothetical fire origin)
    center_lat, center_lon = 40.7128, -74.0060  # Example coordinates
    
    colors = ['orange', 'red', 'darkred']
    opacities = [0.3, 0.5, 0.7]
    
    # Add circles for each time prediction
    for i, (time_key, prediction) in enumerate(spread_predictions.items()):
        radius = prediction['spread_radius']  # in meters
        
        # Convert radius to approximate degrees (rough approximation)
        radius_deg = radius / 111000  # 1 degree ≈ 111km
        
        # Create circle points
        angles = np.linspace(0, 2*np.pi, 50)
        circle_lats = center_lat + radius_deg * np.cos(angles)
        circle_lons = center_lon + radius_deg * np.sin(angles)
        
        fig.add_trace(go.Scattermapbox(
            lat=circle_lats,
            lon=circle_lons,
            mode='lines',
            fill='toself',
            fillcolor=colors[i],
            opacity=opacities[i],
            line=dict(width=2, color=colors[i]),
            name=f'{time_key} Spread',
            hovertemplate=f'<b>{time_key} Prediction</b><br>' +
                         f'Radius: {radius:.0f}m<br>' +
                         f'Area: {prediction["affected_area"]:.1f} hectares<br>' +
                         f'Risk Level: {prediction["risk_prediction"]["risk_level"]}<br>' +
                         '<extra></extra>'
        ))
    
    # Add center point
    fig.add_trace(go.Scattermapbox(
        lat=[center_lat],
        lon=[center_lon],
        mode='markers',
        marker=dict(size=15, color='red', symbol='fire-station'),
        name='Fire Origin',
        hovertemplate='<b>Fire Origin</b><br>Sensor Location<extra></extra>'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12
        ),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)")
    )
    
    return fig

def create_sensor_timeseries(historical_data, selected_sensors):
    """Create time series plots for selected sensors"""
    fig = make_subplots(
        rows=len(selected_sensors), 
        cols=1,
        subplot_titles=selected_sensors,
        shared_xaxes=True,
        vertical_spacing=0.08
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, sensor in enumerate(selected_sensors):
        if sensor in historical_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=historical_data['timestamp'],
                    y=historical_data[sensor],
                    mode='lines',
                    name=sensor.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'<b>{sensor.replace("_", " ").title()}</b><br>' +
                                 '%{y:.2f}<br>' +
                                 '%{x}<br>' +
                                 '<extra></extra>'
                ),
                row=i+1, col=1
            )
    
    fig.update_layout(
        height=150 * len(selected_sensors),
        showlegend=False,
        title="Sensor Data Trends",
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time", row=len(selected_sensors), col=1)
    
    return fig

def create_correlation_heatmap(data):
    """Create correlation heatmap for sensor data"""
    # Select numeric columns only
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'timestamp' in numeric_columns:
        numeric_columns.remove('timestamp')
    
    correlation_matrix = data[numeric_columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Sensor Data Correlation Matrix",
        height=500,
        width=500
    )
    
    return fig

def create_feature_importance_chart(feature_importance):
    """Create feature importance bar chart"""
    features, importance = zip(*feature_importance)
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='darkgreen',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Feature Importance in Fire Risk Prediction",
        xaxis_title="Importance Score",
        yaxis_title="Sensor Features",
        height=400,
        margin=dict(l=150, r=20, t=60, b=20)
    )
    
    return fig

def create_risk_distribution_pie(risk_history):
    """Create pie chart showing risk level distribution"""
    if len(risk_history) == 0:
        return None
    
    risk_counts = pd.Series([r['risk_level'] for r in risk_history]).value_counts()
    
    colors = {'LOW': '#28a745', 'MEDIUM': '#ffc107', 'HIGH': '#dc3545'}
    pie_colors = [colors.get(level, '#6c757d') for level in risk_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=.3,
        marker_colors=pie_colors,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Risk Level Distribution",
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_environmental_radar(current_data):
    """Create radar chart for current environmental conditions"""
    # Normalize values to 0-1 scale for radar chart
    params = [
        ('Temperature', current_data['temperature'] / 50),
        ('Humidity', current_data['humidity'] / 100),
        ('Soil Moisture', current_data['soil_moisture'] / 100),
        ('Wind Speed', current_data['wind_speed'] / 50),
        ('Smoke Level', min(current_data['smoke_level'] / 200, 1)),
        ('Sunlight', current_data['sunlight_intensity'] / 100000),
        ('Pressure', (current_data['barometric_pressure'] - 950) / 100)
    ]
    
    categories = [p[0] for p in params]
    values = [p[1] for p in params]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],  # Close the polygon
        fill='toself',
        fillcolor='rgba(255, 99, 132, 0.2)',
        line=dict(color='rgba(255, 99, 132, 1)', width=2),
        name='Current Conditions'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Current Environmental Conditions",
        height=400
    )
    
    return fig
