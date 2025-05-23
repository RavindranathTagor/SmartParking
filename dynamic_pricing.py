import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime
import pandas as pd

class DynamicPricingModel:
    def __init__(self):
        self.model = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=2)
        
    def prepare_features(self, data):
        """Convert raw features into polynomial features"""
        return self.poly_features.fit_transform(data)
    
    def train(self, X, y):
        """Train the model with historical data"""
        X_poly = self.prepare_features(X)
        self.model.fit(X_poly, y)
        
    def predict_price(self, features):
        """Predict parking price based on current conditions"""
        features_poly = self.poly_features.transform(features.reshape(1, -1))
        return self.model.predict(features_poly)[0]
    
    def get_time_features(self):
        """Extract time-based features"""
        now = datetime.datetime.now()
        hour = now.hour
        is_weekend = 1 if now.weekday() >= 5 else 0
        is_peak_hour = 1 if (8 <= hour <= 10) or (16 <= hour <= 19) else 0
        return hour, is_weekend, is_peak_hour

def generate_sample_data(n_samples=1000):
    """Generate sample training data"""
    np.random.seed(42)
    
    # Features
    distances = np.random.uniform(0, 10, n_samples)  # Distance from city center in km
    traffic_levels = np.random.uniform(0, 100, n_samples)  # Traffic density percentage
    hours = np.random.randint(0, 24, n_samples)
    is_weekend = np.random.randint(0, 2, n_samples)
    available_slots = np.random.randint(0, 100, n_samples)
    
    # Create feature matrix
    X = np.column_stack([
        distances,
        traffic_levels,
        hours,
        is_weekend,
        available_slots
    ])
      # Generate prices with some realistic relationships
    base_price = 50
    prices = (base_price + 
             5 * (10 - distances) +    # Higher price closer to center
             0.3 * traffic_levels +    # Higher price with more traffic
             10 * (is_weekend) +       # Weekend premium
             -0.3 * available_slots)   # Lower price when more slots available
    
    # Add peak hour effects
    peak_hours_morning = (hours >= 8) & (hours <= 10)
    peak_hours_evening = (hours >= 16) & (hours <= 19)
    prices[peak_hours_morning | peak_hours_evening] += 20
    
    # Add some noise
    prices += np.random.normal(0, 5, n_samples)
    prices = np.maximum(prices, 20)  # Ensure minimum price
    
    return X, prices

def create_training_data():
    """Create and save training data"""
    X, y = generate_sample_data()
    
    # Create DataFrame
    columns = ['distance', 'traffic_level', 'hour', 'is_weekend', 'available_slots']
    df = pd.DataFrame(X, columns=columns)
    df['price'] = y
    
    # Save to CSV
    df.to_csv('parking_data.csv', index=False)
    return df
