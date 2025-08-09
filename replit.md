# Forest Fire Monitoring & Prediction System

## Overview

This is a comprehensive forest fire monitoring and prediction system built with Streamlit. The application provides real-time monitoring of environmental conditions, fire risk assessment using machine learning, and fire spread predictions. It features a dashboard interface for visualizing sensor data, historical analysis capabilities, and predictive modeling to help forestry professionals assess and respond to fire risks.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Layout**: Wide layout with expandable sidebar for controls
- **Components**: Modular component-based architecture with separate modules for dashboard, predictions, and historical analysis
- **Visualization**: Plotly for interactive charts, gauges, and maps
- **State Management**: Streamlit session state for maintaining application data across user interactions

### Backend Architecture
- **Data Processing**: Pandas for data manipulation and NumPy for numerical computations
- **Machine Learning**: Scikit-learn based RandomForestClassifier for fire risk prediction
- **Data Generation**: Custom sensor data simulation system with realistic environmental patterns
- **Model Training**: Automated synthetic data generation for training fire risk prediction models

### Data Management
- **Data Storage**: In-memory storage using Pandas DataFrames and Streamlit session state
- **Data Generation**: Real-time sensor data simulation with configurable weather patterns
- **Historical Data**: Accumulative data storage for trend analysis and historical reporting
- **Data Export**: CSV export functionality for historical data analysis

### Prediction System
- **Fire Risk Model**: Machine learning model using environmental factors (temperature, humidity, soil moisture, smoke levels, etc.)
- **Risk Classification**: Three-tier risk system (LOW, MEDIUM, HIGH) with probability scores
- **Fire Spread Prediction**: Spatial modeling for predicting fire spread radius and affected areas
- **Feature Engineering**: Environmental sensor data preprocessing and normalization

### Visualization Components
- **Real-time Dashboard**: Live metrics display with status indicators and alerts
- **Risk Gauges**: Interactive gauge charts for fire risk visualization
- **Fire Spread Maps**: Geographic visualization of predicted fire spread patterns
- **Time Series Charts**: Historical trend analysis for environmental conditions
- **Correlation Analysis**: Heatmaps for understanding relationships between environmental factors

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework for the main interface
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing for data processing
- **plotly**: Interactive visualization library for charts and maps
- **scikit-learn**: Machine learning library for fire risk prediction models
- **joblib**: Model serialization and persistence

### Data Sources
- **Simulated Sensors**: Custom data generator for environmental monitoring (temperature, humidity, barometric pressure, soil moisture, smoke levels, sunlight intensity, wind speed)
- **Manual Input**: User interface for manual data entry and testing
- **Weather Patterns**: Configurable weather simulation with multiple environmental scenarios

### Visualization Dependencies
- **Plotly Express**: High-level plotting interface
- **Plotly Graph Objects**: Low-level plotting for custom visualizations
- **Plotly Subplots**: Multi-panel chart creation for complex dashboards