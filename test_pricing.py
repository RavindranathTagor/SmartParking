import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dynamic_pricing import DynamicPricingModel, create_training_data
import numpy as np

# Get training data
df = create_training_data()

# Initialize and train model
model = DynamicPricingModel()
X = df[['distance', 'traffic_level', 'hour', 'is_weekend', 'available_slots']].values
y = df['price'].values
model.train(X, y)

# Create visualizations
plt.figure(figsize=(15, 10))

# 1. Price vs Distance from City Center
plt.subplot(2, 2, 1)
plt.scatter(df['distance'], df['price'], alpha=0.5)
plt.xlabel('Distance from City Center (km)')
plt.ylabel('Price (₹)')
plt.title('Price vs Distance from City Center')

# 2. Price vs Time of Day
plt.subplot(2, 2, 2)
plt.scatter(df['hour'], df['price'], alpha=0.5)
plt.xlabel('Hour of Day')
plt.ylabel('Price (₹)')
plt.title('Price vs Time of Day')

# 3. Price vs Available Slots
plt.subplot(2, 2, 3)
plt.scatter(df['available_slots'], df['price'], alpha=0.5)
plt.xlabel('Available Slots')
plt.ylabel('Price (₹)')
plt.title('Price vs Available Slots')

# 4. Average Price by Weekend vs Weekday
plt.subplot(2, 2, 4)
weekend_avg = df.groupby('is_weekend')['price'].mean()
weekend_avg.plot(kind='bar')
plt.xticks([0, 1], ['Weekday', 'Weekend'])
plt.ylabel('Average Price (₹)')
plt.title('Average Price: Weekday vs Weekend')

plt.tight_layout()
plt.savefig('pricing_analysis.png')

# Print some sample predictions
print("\nSample Price Predictions:")
print("=" * 50)

# Test different scenarios
scenarios = [
    {
        'distance': 2,
        'traffic_level': 80,
        'hour': 9,
        'is_weekend': 1,
        'available_slots': 10,
        'description': 'Peak hour, weekend, near city center, high traffic, low availability'
    },
    {
        'distance': 8,
        'traffic_level': 20,
        'hour': 14,
        'is_weekend': 0,
        'available_slots': 80,
        'description': 'Off-peak hour, weekday, far from city center, low traffic, high availability'
    },
    {
        'distance': 5,
        'traffic_level': 50,
        'hour': 18,
        'is_weekend': 0,
        'available_slots': 40,
        'description': 'Evening peak, weekday, medium distance, medium traffic, medium availability'
    }
]

for scenario in scenarios:
    features = np.array([
        scenario['distance'],
        scenario['traffic_level'],
        scenario['hour'],
        scenario['is_weekend'],
        scenario['available_slots']
    ])
    price = model.predict_price(features)
    print(f"\nScenario: {scenario['description']}")
    print(f"Predicted Price: ₹{price:.2f}/hr")

# Save some sample data to CSV for inspection
df.head(20).to_csv('sample_pricing_data.csv', index=False)
print("\nData files generated:")
print("- pricing_analysis.png: Visualizations of pricing factors")
print("- sample_pricing_data.csv: Sample of training data")
print("- parking_data.csv: Full training dataset")
