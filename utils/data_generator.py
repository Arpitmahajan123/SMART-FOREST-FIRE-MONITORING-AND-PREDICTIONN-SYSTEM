import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class SensorDataGenerator:
    def __init__(self):
        # Define realistic ranges and patterns for each sensor
        self.sensor_config = {
            'temperature': {'base': 25, 'range': 15, 'daily_variation': 8},
            'humidity': {'base': 60, 'range': 30, 'daily_variation': 20},
            'barometric_pressure': {'base': 1013.25, 'range': 30, 'daily_variation': 5},
            'soil_moisture': {'base': 45, 'range': 35, 'daily_variation': 10},
            'smoke_level': {'base': 50, 'range': 100, 'daily_variation': 30},
            'sunlight_intensity': {'base': 40000, 'range': 50000, 'daily_variation': 35000},
            'wind_speed': {'base': 12, 'range': 15, 'daily_variation': 8}
        }
        
        # Weather patterns
        self.weather_patterns = ['sunny', 'cloudy', 'windy', 'dry', 'humid']
        self.current_pattern = random.choice(self.weather_patterns)
        
        # Base time for simulation
        self.base_time = datetime.now()
    
    def generate_current_reading(self):
        """Generate current sensor readings"""
        current_time = datetime.now()
        hour_of_day = current_time.hour
        
        # Apply daily cycles
        temp_cycle = np.sin((hour_of_day - 6) * np.pi / 12) * 0.5 + 0.5  # Peak at 2 PM
        sun_cycle = max(0, np.sin((hour_of_day - 6) * np.pi / 12))  # No sun at night
        
        # Generate base readings
        data = {}
        
        # Temperature (higher during day, affected by sunlight)
        base_temp = self.sensor_config['temperature']['base']
        temp_variation = self.sensor_config['temperature']['daily_variation']
        data['temperature'] = base_temp + (temp_cycle - 0.5) * temp_variation + np.random.normal(0, 2)
        
        # Humidity (inversely related to temperature)
        base_humidity = self.sensor_config['humidity']['base']
        humidity_variation = self.sensor_config['humidity']['daily_variation']
        temp_effect = (data['temperature'] - base_temp) * -0.8  # Inverse relationship
        data['humidity'] = base_humidity + temp_effect + np.random.normal(0, 5)
        data['humidity'] = max(10, min(95, data['humidity']))  # Realistic bounds
        
        # Barometric pressure (slight daily variation)
        base_pressure = self.sensor_config['barometric_pressure']['base']
        data['barometric_pressure'] = base_pressure + np.random.normal(0, 3) + np.sin(hour_of_day * np.pi / 12) * 2
        
        # Soil moisture (affected by temperature and humidity)
        base_soil = self.sensor_config['soil_moisture']['base']
        temp_soil_effect = (data['temperature'] - base_temp) * -0.5
        humidity_soil_effect = (data['humidity'] - base_humidity) * 0.3
        data['soil_moisture'] = base_soil + temp_soil_effect + humidity_soil_effect + np.random.normal(0, 3)
        data['soil_moisture'] = max(5, min(90, data['soil_moisture']))
        
        # Smoke level (baseline with occasional spikes)
        base_smoke = self.sensor_config['smoke_level']['base']
        spike_chance = 0.05  # 5% chance of smoke spike
        if np.random.random() < spike_chance:
            data['smoke_level'] = base_smoke + np.random.exponential(100)
        else:
            data['smoke_level'] = base_smoke + np.random.normal(0, 10)
        data['smoke_level'] = max(0, min(1000, data['smoke_level']))
        
        # Sunlight intensity (follows solar cycle)
        base_sun = self.sensor_config['sunlight_intensity']['base']
        data['sunlight_intensity'] = base_sun * sun_cycle + np.random.normal(0, 2000)
        data['sunlight_intensity'] = max(0, min(100000, data['sunlight_intensity']))
        
        # Wind speed (varies throughout day with weather pattern influence)
        base_wind = self.sensor_config['wind_speed']['base']
        pattern_effect = self._get_weather_pattern_effect()
        data['wind_speed'] = base_wind + pattern_effect['wind'] + np.random.normal(0, 2)
        data['wind_speed'] = max(0, min(80, data['wind_speed']))
        
        # Apply weather pattern effects
        self._apply_weather_pattern(data, pattern_effect)
        
        # Add timestamp
        data['timestamp'] = current_time
        
        return data
    
    def _get_weather_pattern_effect(self):
        """Get effects based on current weather pattern"""
        effects = {
            'sunny': {'temp': 3, 'humidity': -10, 'wind': 2, 'sun': 15000},
            'cloudy': {'temp': -2, 'humidity': 5, 'wind': -1, 'sun': -20000},
            'windy': {'temp': -1, 'humidity': -5, 'wind': 8, 'sun': 5000},
            'dry': {'temp': 5, 'humidity': -15, 'wind': 3, 'sun': 10000},
            'humid': {'temp': -3, 'humidity': 15, 'wind': -2, 'sun': -5000}
        }
        return effects.get(self.current_pattern, {'temp': 0, 'humidity': 0, 'wind': 0, 'sun': 0})
    
    def _apply_weather_pattern(self, data, pattern_effect):
        """Apply weather pattern effects to sensor data"""
        data['temperature'] += pattern_effect['temp']
        data['humidity'] += pattern_effect['humidity']
        data['wind_speed'] += pattern_effect['wind']
        data['sunlight_intensity'] += pattern_effect['sun']
        
        # Ensure bounds
        data['temperature'] = max(-10, min(50, data['temperature']))
        data['humidity'] = max(10, min(95, data['humidity']))
        data['wind_speed'] = max(0, min(80, data['wind_speed']))
        data['sunlight_intensity'] = max(0, min(100000, data['sunlight_intensity']))
        
        # Randomly change weather pattern
        if np.random.random() < 0.1:  # 10% chance to change pattern
            self.current_pattern = random.choice(self.weather_patterns)
    
    def generate_historical_data(self, hours=24, interval_minutes=15):
        """Generate historical sensor data"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Generate time series
        time_points = []
        current = start_time
        while current <= end_time:
            time_points.append(current)
            current += timedelta(minutes=interval_minutes)
        
        historical_data = []
        
        for timestamp in time_points:
            # Temporarily adjust base time for historical generation
            hour_of_day = timestamp.hour
            
            # Generate data for this timestamp
            data = self._generate_historical_point(timestamp, hour_of_day)
            data['timestamp'] = timestamp
            historical_data.append(data)
        
        return pd.DataFrame(historical_data)
    
    def _generate_historical_point(self, timestamp, hour_of_day):
        """Generate a single historical data point"""
        # Apply daily cycles
        temp_cycle = np.sin((hour_of_day - 6) * np.pi / 12) * 0.5 + 0.5
        sun_cycle = max(0, np.sin((hour_of_day - 6) * np.pi / 12))
        
        data = {}
        
        # Temperature
        base_temp = self.sensor_config['temperature']['base']
        temp_variation = self.sensor_config['temperature']['daily_variation']
        seasonal_effect = np.sin((timestamp.timetuple().tm_yday - 80) * 2 * np.pi / 365) * 5  # Seasonal variation
        data['temperature'] = base_temp + (temp_cycle - 0.5) * temp_variation + seasonal_effect + np.random.normal(0, 2)
        
        # Humidity
        base_humidity = self.sensor_config['humidity']['base']
        temp_effect = (data['temperature'] - base_temp) * -0.8
        data['humidity'] = base_humidity + temp_effect + np.random.normal(0, 5)
        data['humidity'] = max(10, min(95, data['humidity']))
        
        # Barometric pressure
        base_pressure = self.sensor_config['barometric_pressure']['base']
        data['barometric_pressure'] = base_pressure + np.random.normal(0, 3) + np.sin(hour_of_day * np.pi / 12) * 2
        
        # Soil moisture
        base_soil = self.sensor_config['soil_moisture']['base']
        temp_soil_effect = (data['temperature'] - base_temp) * -0.5
        humidity_soil_effect = (data['humidity'] - base_humidity) * 0.3
        data['soil_moisture'] = base_soil + temp_soil_effect + humidity_soil_effect + np.random.normal(0, 3)
        data['soil_moisture'] = max(5, min(90, data['soil_moisture']))
        
        # Smoke level
        base_smoke = self.sensor_config['smoke_level']['base']
        spike_chance = 0.02  # Lower chance for historical data
        if np.random.random() < spike_chance:
            data['smoke_level'] = base_smoke + np.random.exponential(80)
        else:
            data['smoke_level'] = base_smoke + np.random.normal(0, 8)
        data['smoke_level'] = max(0, min(1000, data['smoke_level']))
        
        # Sunlight intensity
        base_sun = self.sensor_config['sunlight_intensity']['base']
        data['sunlight_intensity'] = base_sun * sun_cycle + np.random.normal(0, 2000)
        data['sunlight_intensity'] = max(0, min(100000, data['sunlight_intensity']))
        
        # Wind speed
        base_wind = self.sensor_config['wind_speed']['base']
        data['wind_speed'] = base_wind + np.random.normal(0, 3)
        data['wind_speed'] = max(0, min(80, data['wind_speed']))
        
        return data
