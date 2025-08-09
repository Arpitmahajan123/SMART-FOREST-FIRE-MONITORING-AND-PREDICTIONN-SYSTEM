import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils.visualization import create_sensor_timeseries, create_correlation_heatmap
from datetime import datetime, timedelta

def render_historical_analysis(historical_data):
    """Render the historical data analysis section"""
    
    st.subheader("📈 Historical Data Analysis")
    
    if historical_data.empty:
        st.warning("No historical data available yet. Data will accumulate as the system runs.")
        return
    
    # Time range selector
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_range = st.selectbox(
            "📅 Time Range",
            ["Last 1 Hour", "Last 6 Hours", "Last 12 Hours", "Last 24 Hours", "All Data"],
            index=3
        )
    
    with col2:
        # Data aggregation option
        aggregation = st.selectbox(
            "📊 Data Aggregation",
            ["Raw Data", "5-minute Average", "15-minute Average", "Hourly Average"],
            index=1
        )
    
    with col3:
        # Export data option
        if st.button("💾 Export Historical Data"):
            csv = historical_data.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"forest_fire_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Filter data based on time range
    filtered_data = filter_data_by_time_range(historical_data, time_range)
    
    # Aggregate data if requested
    if aggregation != "Raw Data":
        filtered_data = aggregate_data(filtered_data, aggregation)
    
    if filtered_data.empty:
        st.warning("No data available for the selected time range.")
        return
    
    st.markdown("---")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Data Points",
            len(filtered_data),
            f"{len(filtered_data) - len(historical_data)} new"
        )
    
    with col2:
        avg_temp = filtered_data['temperature'].mean()
        st.metric(
            "🌡️ Avg Temperature",
            f"{avg_temp:.1f}°C",
            f"{avg_temp - 25:.1f}°C from normal"
        )
    
    with col3:
        avg_humidity = filtered_data['humidity'].mean()
        st.metric(
            "💧 Avg Humidity",
            f"{avg_humidity:.1f}%",
            f"{avg_humidity - 60:.1f}% from normal"
        )
    
    with col4:
        max_smoke = filtered_data['smoke_level'].max()
        if max_smoke > 200:
            st.metric("🚨 Max Smoke", f"{max_smoke:.0f} ppm", "⚠️ High levels detected")
        else:
            st.metric("💨 Max Smoke", f"{max_smoke:.0f} ppm", "✅ Normal levels")
    
    # Main analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🔗 Correlations", "📊 Distributions", "🎯 Risk Analysis"])
    
    with tab1:
        render_trends_analysis(filtered_data)
    
    with tab2:
        render_correlation_analysis(filtered_data)
    
    with tab3:
        render_distribution_analysis(filtered_data)
    
    with tab4:
        render_risk_analysis(filtered_data)

def filter_data_by_time_range(data, time_range):
    """Filter data based on selected time range"""
    if data.empty or time_range == "All Data":
        return data
    
    now = datetime.now()
    
    if time_range == "Last 1 Hour":
        cutoff = now - timedelta(hours=1)
    elif time_range == "Last 6 Hours":
        cutoff = now - timedelta(hours=6)
    elif time_range == "Last 12 Hours":
        cutoff = now - timedelta(hours=12)
    elif time_range == "Last 24 Hours":
        cutoff = now - timedelta(hours=24)
    else:
        return data
    
    return data[data['timestamp'] >= cutoff]

def aggregate_data(data, aggregation):
    """Aggregate data based on selected method"""
    if data.empty or aggregation == "Raw Data":
        return data
    
    # Set timestamp as index for resampling
    data = data.copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    
    # Define resampling frequency
    freq_map = {
        "5-minute Average": "5min",
        "15-minute Average": "15min", 
        "Hourly Average": "1H"
    }
    
    freq = freq_map.get(aggregation, "5T")
    
    # Resample and aggregate
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    aggregated = data[numeric_columns].resample(freq).mean()
    
    # Reset index
    aggregated.reset_index(inplace=True)
    
    return aggregated

def render_trends_analysis(data):
    """Render trends analysis section"""
    st.subheader("📈 Sensor Data Trends")
    
    # Sensor selection
    available_sensors = [col for col in data.columns if col != 'timestamp']
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_sensors = st.multiselect(
            "Select Sensors to Display:",
            available_sensors,
            default=available_sensors[:4] if len(available_sensors) >= 4 else available_sensors,
            help="Choose which sensors to display in the trend chart"
        )
        
        # Chart type selection
        chart_type = st.radio(
            "Chart Type:",
            ["Line Chart", "Area Chart", "Scatter Plot"],
            help="Select visualization type"
        )
        
        # Show statistical info
        if selected_sensors:
            st.markdown("### 📊 Statistics")
            for sensor in selected_sensors[:3]:  # Show stats for first 3 selected sensors
                if sensor in data.columns:
                    sensor_data = data[sensor]
                    st.write(f"**{sensor.replace('_', ' ').title()}:**")
                    st.write(f"  Mean: {sensor_data.mean():.2f}")
                    st.write(f"  Std: {sensor_data.std():.2f}")
                    st.write(f"  Min: {sensor_data.min():.2f}")
                    st.write(f"  Max: {sensor_data.max():.2f}")
    
    with col2:
        if selected_sensors:
            # Create time series chart
            if chart_type == "Line Chart":
                fig = create_multi_line_chart(data, selected_sensors)
            elif chart_type == "Area Chart":
                fig = create_area_chart(data, selected_sensors)
            else:
                fig = create_scatter_plot(data, selected_sensors)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trend analysis insights
            st.markdown("### 🔍 Trend Insights")
            analyze_trends(data, selected_sensors)
        else:
            st.info("Please select at least one sensor to display trends.")

def render_correlation_analysis(data):
    """Render correlation analysis section"""
    st.subheader("🔗 Sensor Correlation Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Correlation heatmap
        correlation_chart = create_correlation_heatmap(data)
        st.plotly_chart(correlation_chart, use_container_width=True)
    
    with col2:
        # Correlation insights
        st.markdown("### 🔍 Correlation Insights")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'timestamp' in numeric_columns:
            numeric_columns.remove('timestamp')
        
        if len(numeric_columns) >= 2:
            correlation_matrix = data[numeric_columns].corr()
            
            # Find strongest correlations
            correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    correlations.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_val
                    ))
            
            # Sort by absolute correlation value
            correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            st.markdown("**Strongest Correlations:**")
            for i, (var1, var2, corr) in enumerate(correlations[:5]):
                correlation_strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
                correlation_direction = "Positive" if corr > 0 else "Negative"
                
                st.write(f"{i+1}. **{var1.replace('_', ' ').title()}** vs **{var2.replace('_', ' ').title()}**")
                st.write(f"   {correlation_direction} {correlation_strength} correlation ({corr:.3f})")
                
                if abs(corr) > 0.6:
                    if corr > 0:
                        st.write("   ↗️ These parameters tend to increase together")
                    else:
                        st.write("   ↘️ When one increases, the other tends to decrease")
            
            # Scatter plot for strongest correlation
            if correlations:
                st.markdown("---")
                st.markdown("### 📊 Scatter Plot - Strongest Correlation")
                strongest_corr = correlations[0]
                var1, var2, corr_val = strongest_corr
                
                fig = px.scatter(
                    data, 
                    x=var1, 
                    y=var2,
                    title=f"{var1.replace('_', ' ').title()} vs {var2.replace('_', ' ').title()} (r={corr_val:.3f})",
                    trendline="ols"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

def render_distribution_analysis(data):
    """Render distribution analysis section"""
    st.subheader("📊 Data Distributions")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'timestamp' in numeric_columns:
        numeric_columns.remove('timestamp')
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_param = st.selectbox(
            "Select Parameter:",
            numeric_columns,
            help="Choose parameter to analyze distribution"
        )
        
        analysis_type = st.radio(
            "Analysis Type:",
            ["Histogram", "Box Plot", "Time-based Distribution"],
            help="Select type of distribution analysis"
        )
        
        # Statistical summary
        if selected_param in data.columns:
            param_data = data[selected_param]
            st.markdown(f"### 📈 {selected_param.replace('_', ' ').title()} Statistics")
            st.write(f"**Count:** {len(param_data)}")
            st.write(f"**Mean:** {param_data.mean():.3f}")
            st.write(f"**Median:** {param_data.median():.3f}")
            st.write(f"**Std Dev:** {param_data.std():.3f}")
            st.write(f"**Min:** {param_data.min():.3f}")
            st.write(f"**Max:** {param_data.max():.3f}")
            
            # Quartiles
            q25 = param_data.quantile(0.25)
            q75 = param_data.quantile(0.75)
            st.write(f"**Q25:** {q25:.3f}")
            st.write(f"**Q75:** {q75:.3f}")
            st.write(f"**IQR:** {q75 - q25:.3f}")
    
    with col2:
        if selected_param in data.columns:
            if analysis_type == "Histogram":
                fig = px.histogram(
                    data, 
                    x=selected_param,
                    nbins=30,
                    title=f"{selected_param.replace('_', ' ').title()} Distribution",
                    marginal="box"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_type == "Box Plot":
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=data[selected_param],
                    name=selected_param.replace('_', ' ').title(),
                    boxpoints='outliers'
                ))
                fig.update_layout(
                    title=f"{selected_param.replace('_', ' ').title()} Box Plot",
                    yaxis_title=selected_param.replace('_', ' ').title(),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # Time-based Distribution
                data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
                hourly_stats = data.groupby('hour')[selected_param].agg(['mean', 'std', 'min', 'max']).reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hourly_stats['hour'],
                    y=hourly_stats['mean'],
                    mode='lines+markers',
                    name='Mean',
                    line=dict(color='blue', width=2)
                ))
                
                # Add confidence band
                fig.add_trace(go.Scatter(
                    x=hourly_stats['hour'],
                    y=hourly_stats['mean'] + hourly_stats['std'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=hourly_stats['hour'],
                    y=hourly_stats['mean'] - hourly_stats['std'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)',
                    name='±1 Std Dev',
                    hoverinfo='skip'
                ))
                
                fig.update_layout(
                    title=f"{selected_param.replace('_', ' ').title()} by Hour of Day",
                    xaxis_title="Hour of Day",
                    yaxis_title=selected_param.replace('_', ' ').title(),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

def render_risk_analysis(data):
    """Render risk analysis section"""
    st.subheader("🎯 Historical Risk Analysis")
    
    # Since we don't have historical risk predictions, we'll create some analysis
    # based on the current risk factors
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔥 Risk Factor Analysis")
        
        # Create risk indicators based on thresholds
        risk_factors = []
        
        if 'temperature' in data.columns:
            high_temp_periods = (data['temperature'] > 35).sum()
            risk_factors.append(('High Temperature', high_temp_periods, len(data)))
        
        if 'humidity' in data.columns:
            low_humidity_periods = (data['humidity'] < 30).sum()
            risk_factors.append(('Low Humidity', low_humidity_periods, len(data)))
        
        if 'wind_speed' in data.columns:
            high_wind_periods = (data['wind_speed'] > 25).sum()
            risk_factors.append(('High Wind', high_wind_periods, len(data)))
        
        if 'smoke_level' in data.columns:
            high_smoke_periods = (data['smoke_level'] > 200).sum()
            risk_factors.append(('Elevated Smoke', high_smoke_periods, len(data)))
        
        # Display risk factor statistics
        for factor, count, total in risk_factors:
            percentage = (count / total) * 100 if total > 0 else 0
            
            if percentage > 20:
                st.error(f"**{factor}:** {count}/{total} periods ({percentage:.1f}%)")
            elif percentage > 10:
                st.warning(f"**{factor}:** {count}/{total} periods ({percentage:.1f}%)")
            else:
                st.success(f"**{factor}:** {count}/{total} periods ({percentage:.1f}%)")
    
    with col2:
        st.markdown("### 📊 Risk Timeline")
        
        # Create a composite risk score
        if len(data) > 0:
            risk_score = calculate_composite_risk_score(data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=risk_score,
                mode='lines',
                name='Risk Score',
                line=dict(color='red', width=2),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)'
            ))
            
            # Add threshold lines
            fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                         annotation_text="High Risk Threshold")
            fig.add_hline(y=0.3, line_dash="dash", line_color="orange", 
                         annotation_text="Medium Risk Threshold")
            
            fig.update_layout(
                title="Historical Risk Score Timeline",
                xaxis_title="Time",
                yaxis_title="Risk Score",
                height=300,
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk pattern analysis
    if len(data) > 0:
        st.markdown("---")
        st.markdown("### 🕒 Risk Patterns by Time of Day")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly risk analysis
            data_copy = data.copy()
            data_copy['hour'] = pd.to_datetime(data_copy['timestamp']).dt.hour
            data_copy['risk_score'] = calculate_composite_risk_score(data_copy)
            
            hourly_risk = data_copy.groupby('hour')['risk_score'].mean().reset_index()
            
            fig = px.bar(
                hourly_risk,
                x='hour',
                y='risk_score',
                title="Average Risk Score by Hour",
                color='risk_score',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Daily pattern insights
            st.markdown("**Time-Based Risk Insights:**")
            
            peak_hour = hourly_risk.loc[hourly_risk['risk_score'].idxmax(), 'hour']
            lowest_hour = hourly_risk.loc[hourly_risk['risk_score'].idxmin(), 'hour']
            
            st.write(f"🔥 **Peak Risk Hour:** {peak_hour}:00")
            st.write(f"✅ **Lowest Risk Hour:** {lowest_hour}:00")
            
            morning_risk = hourly_risk[(hourly_risk['hour'] >= 6) & (hourly_risk['hour'] < 12)]['risk_score'].mean()
            afternoon_risk = hourly_risk[(hourly_risk['hour'] >= 12) & (hourly_risk['hour'] < 18)]['risk_score'].mean()
            evening_risk = hourly_risk[(hourly_risk['hour'] >= 18) & (hourly_risk['hour'] < 24)]['risk_score'].mean()
            night_risk = hourly_risk[(hourly_risk['hour'] >= 0) & (hourly_risk['hour'] < 6)]['risk_score'].mean()
            
            st.write(f"🌅 **Morning Risk:** {morning_risk:.2f}")
            st.write(f"☀️ **Afternoon Risk:** {afternoon_risk:.2f}")
            st.write(f"🌆 **Evening Risk:** {evening_risk:.2f}")
            st.write(f"🌙 **Night Risk:** {night_risk:.2f}")
            
            # Risk recommendations
            st.markdown("**📋 Recommendations:**")
            if afternoon_risk > 0.6:
                st.warning("⚠️ Increase afternoon monitoring")
            if peak_hour >= 12 and peak_hour <= 16:
                st.info("🔥 Peak fire risk during mid-day hours")
            if night_risk < 0.3:
                st.success("✅ Night time shows lower fire risk")

def calculate_composite_risk_score(data):
    """Calculate a composite risk score based on multiple factors"""
    risk_score = np.zeros(len(data))
    
    # Temperature contribution (normalized to 0-1)
    if 'temperature' in data.columns:
        temp_risk = np.clip((data['temperature'] - 20) / 20, 0, 1)  # Risk increases above 20°C
        risk_score += temp_risk * 0.25
    
    # Humidity contribution (inverse relationship)
    if 'humidity' in data.columns:
        humidity_risk = np.clip((80 - data['humidity']) / 60, 0, 1)  # Risk increases below 80%
        risk_score += humidity_risk * 0.25
    
    # Wind speed contribution
    if 'wind_speed' in data.columns:
        wind_risk = np.clip(data['wind_speed'] / 40, 0, 1)  # Risk increases with wind
        risk_score += wind_risk * 0.2
    
    # Smoke level contribution
    if 'smoke_level' in data.columns:
        smoke_risk = np.clip(data['smoke_level'] / 300, 0, 1)  # Risk increases with smoke
        risk_score += smoke_risk * 0.3
    
    return np.clip(risk_score, 0, 1)

def create_multi_line_chart(data, sensors):
    """Create multi-line time series chart"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, sensor in enumerate(sensors):
        if sensor in data.columns:
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data[sensor],
                mode='lines',
                name=sensor.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{sensor.replace("_", " ").title()}</b><br>' +
                             '%{y:.2f}<br>' +
                             '%{x}<br>' +
                             '<extra></extra>'
            ))
    
    fig.update_layout(
        title="Sensor Data Trends Over Time",
        xaxis_title="Time",
        yaxis_title="Values",
        height=500,
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)")
    )
    
    return fig

def create_area_chart(data, sensors):
    """Create stacked area chart"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, sensor in enumerate(sensors):
        if sensor in data.columns:
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data[sensor],
                mode='lines',
                name=sensor.replace('_', ' ').title(),
                fill='tonexty' if i > 0 else 'tozeroy',
                fillcolor=colors[i % len(colors)],
                line=dict(color=colors[i % len(colors)], width=0)
            ))
    
    fig.update_layout(
        title="Sensor Data - Stacked Area View",
        xaxis_title="Time",
        yaxis_title="Values",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_scatter_plot(data, sensors):
    """Create scatter plot for selected sensors"""
    if len(sensors) >= 2:
        fig = px.scatter(
            data,
            x=sensors[0],
            y=sensors[1],
            color='timestamp' if len(sensors) == 2 else sensors[2] if len(sensors) > 2 else None,
            title=f"{sensors[0].replace('_', ' ').title()} vs {sensors[1].replace('_', ' ').title()}",
            hover_data=['timestamp']
        )
        fig.update_layout(height=500)
        return fig
    else:
        return go.Figure().add_annotation(text="Please select at least 2 sensors for scatter plot", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

def analyze_trends(data, sensors):
    """Analyze and display trend insights"""
    insights = []
    
    for sensor in sensors:
        if sensor in data.columns and len(data) > 1:
            values = data[sensor].values
            
            # Simple trend calculation
            if len(values) > 1:
                recent_avg = np.mean(values[-min(10, len(values)):])  # Last 10 points or all if less
                overall_avg = np.mean(values)
                
                if recent_avg > overall_avg * 1.1:
                    trend = "📈 Increasing"
                elif recent_avg < overall_avg * 0.9:
                    trend = "📉 Decreasing"
                else:
                    trend = "➡️ Stable"
                
                insights.append(f"**{sensor.replace('_', ' ').title()}:** {trend}")
    
    if insights:
        for insight in insights:
            st.write(insight)
    else:
        st.info("Insufficient data for trend analysis")
