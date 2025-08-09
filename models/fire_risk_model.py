import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime, timedelta

class FireRiskPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'temperature', 'humidity', 'barometric_pressure', 
            'soil_moisture', 'smoke_level', 'sunlight_intensity', 'wind_speed'
        ]
        self.risk_thresholds = {
            'LOW': 0.3,
            'MEDIUM': 0.7,
            'HIGH': 1.0
        }
        self._train_model()
    
    def _generate_training_data(self, n_samples=10000):
        """Generate synthetic training data for the fire risk model"""
        np.random.seed(42)  # For reproducibility
        
        data = []
        
        for _ in range(n_samples):
            # Generate correlated features that make sense for fire risk
            
            # High fire risk scenario (30% of data)
            if np.random.random() < 0.3:
                temperature = np.random.normal(35, 5)  # Higher temperature
                humidity = np.random.normal(25, 8)     # Lower humidity
                soil_moisture = np.random.normal(20, 8)  # Dry soil
                smoke_level = np.random.normal(150, 50)  # Higher smoke
                sunlight_intensity = np.random.normal(75000, 15000)  # Bright sun
                wind_speed = np.random.normal(25, 8)     # Higher wind
                barometric_pressure = np.random.normal(1010, 10)
                fire_risk = 1  # High risk
            
            # Medium fire risk scenario (40% of data)
            elif np.random.random() < 0.7:
                temperature = np.random.normal(28, 6)
                humidity = np.random.normal(45, 12)
                soil_moisture = np.random.normal(35, 10)
                smoke_level = np.random.normal(75, 30)
                sunlight_intensity = np.random.normal(50000, 20000)
                wind_speed = np.random.normal(15, 6)
                barometric_pressure = np.random.normal(1013, 8)
                fire_risk = 1 if np.random.random() < 0.4 else 0  # 40% risk
            
            # Low fire risk scenario (30% of data)
            else:
                temperature = np.random.normal(22, 5)   # Moderate temperature
                humidity = np.random.normal(65, 15)     # Higher humidity
                soil_moisture = np.random.normal(55, 15)  # Moist soil
                smoke_level = np.random.normal(25, 15)   # Low smoke
                sunlight_intensity = np.random.normal(30000, 15000)  # Moderate sun
                wind_speed = np.random.normal(8, 4)      # Low wind
                barometric_pressure = np.random.normal(1015, 10)
                fire_risk = 0  # Low risk
            
            # Ensure realistic ranges
            temperature = max(-10, min(50, temperature))
            humidity = max(0, min(100, humidity))
            soil_moisture = max(0, min(100, soil_moisture))
            smoke_level = max(0, min(1000, smoke_level))
            sunlight_intensity = max(0, min(100000, sunlight_intensity))
            wind_speed = max(0, min(100, wind_speed))
            barometric_pressure = max(950, min(1050, barometric_pressure))
            
            data.append([
                temperature, humidity, barometric_pressure,
                soil_moisture, smoke_level, sunlight_intensity,
                wind_speed, fire_risk
            ])
        
        columns = self.feature_names + ['fire_risk']
        return pd.DataFrame(data, columns=columns)
    
    def _train_model(self):
        """Train the fire risk prediction model"""
        # Generate training data
        training_data = self._generate_training_data()
        
        # Prepare features and target
        X = training_data[self.feature_names]
        y = training_data['fire_risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.classification_report = classification_report(y_test, y_pred)
        
        print(f"Model trained with accuracy: {self.accuracy:.3f}")
    
    def predict_risk(self, sensor_data):
        """Predict fire risk from sensor data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Extract features as DataFrame to maintain feature names
        import pandas as pd
        features_df = pd.DataFrame([sensor_data])[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Get prediction and probability
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Determine risk level
        fire_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        if fire_probability < self.risk_thresholds['LOW']:
            risk_level = 'LOW'
            color = 'green'
        elif fire_probability < self.risk_thresholds['MEDIUM']:
            risk_level = 'MEDIUM'
            color = 'orange'
        else:
            risk_level = 'HIGH'
            color = 'red'
        
        return {
            'prediction': int(prediction),
            'probability': fire_probability,
            'risk_level': risk_level,
            'color': color,
            'confidence': max(probabilities)
        }
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        feature_importance = list(zip(self.feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance
    
    def predict_fire_spread(self, current_data, hours=[1, 2, 3]):
        """Predict fire spread for different time intervals"""
        predictions = {}
        
        for hour in hours:
            # Simulate environmental changes over time
            future_data = current_data.copy()
            
            # Simple heuristic: conditions typically worsen during day, improve at night
            current_hour = datetime.now().hour
            future_hour = (current_hour + hour) % 24
            
            # Day time (6-18): conditions may worsen
            if 6 <= future_hour <= 18:
                future_data['temperature'] += hour * 0.5  # Temperature rises
                future_data['humidity'] -= hour * 1.5     # Humidity drops
                future_data['wind_speed'] += hour * 0.8   # Wind increases
                future_data['sunlight_intensity'] += hour * 2000
            else:
                # Night time: conditions improve
                future_data['temperature'] -= hour * 0.3
                future_data['humidity'] += hour * 2.0
                future_data['wind_speed'] -= hour * 0.5
                future_data['sunlight_intensity'] -= hour * 5000
            
            # Ensure realistic bounds
            future_data['temperature'] = max(-10, min(50, future_data['temperature']))
            future_data['humidity'] = max(0, min(100, future_data['humidity']))
            future_data['wind_speed'] = max(0, min(100, future_data['wind_speed']))
            future_data['sunlight_intensity'] = max(0, min(100000, future_data['sunlight_intensity']))
            
            # Predict risk for future conditions
            future_risk = self.predict_risk(future_data)
            
            predictions[f'{hour}h'] = {
                'risk_prediction': future_risk,
                'environmental_data': future_data,
                'spread_radius': self._calculate_spread_radius(future_risk, hour),
                'affected_area': self._calculate_affected_area(future_risk, hour)
            }
        
        return predictions
    
    def _calculate_spread_radius(self, risk_prediction, hours):
        """Calculate fire spread radius based on risk and time"""
        base_radius = 100  # meters
        
        risk_multiplier = {
            'LOW': 0.5,
            'MEDIUM': 1.0,
            'HIGH': 2.0
        }
        
        multiplier = risk_multiplier.get(risk_prediction['risk_level'], 1.0)
        radius = base_radius * multiplier * np.sqrt(hours)  # Non-linear spread
        
        return min(radius, 5000)  # Cap at 5km
    
    def _calculate_affected_area(self, risk_prediction, hours):
        """Calculate affected area in hectares"""
        radius = self._calculate_spread_radius(risk_prediction, hours)
        area_m2 = np.pi * (radius ** 2)
        area_hectares = area_m2 / 10000  # Convert to hectares
        
        return area_hectares
