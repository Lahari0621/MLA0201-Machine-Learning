import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data Preprocessing - ROBUST sqft parsing
df = pd.read_csv("Bengaluru_House_Data.csv")
df = df[['size', 'total_sqft', 'price']].dropna(subset=['price'])

# Ultra-safe sqft parser (handles EVERY format)
def safe_sqft_to_num(x):
    x = str(x).strip()
    # Remove units and take first number
    import re
    numbers = re.findall(r'\d+\.?\d*', x)
    if numbers:
        return float(numbers[0])
    return 1000  # Default for weird cases

df['total_sqft_num'] = df['total_sqft'].apply(safe_sqft_to_num)

# Safe BHK extraction
df['bhk'] = df['size'].str.extract(r'(\d+)').astype(float).fillna(2)

# Clean outliers
df = df[(df['price'] < 200) & 
        (df['total_sqft_num'] > 300) & 
        (df['total_sqft_num'] < 6000) & 
        (df['bhk'] > 0) & 
        (df['bhk'] < 10)]
df = df.dropna()

print("Cleaned dataset:", df.shape)

# 2. Feature Extraction
X = df[['total_sqft_num', 'bhk']]
y = df['price']

# 3-4. Model Selection + Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

# 5. Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("🏆 RANDOM FOREST HOUSE PRICE PREDICTION")
print("="*60)
print(f"✅ RMSE: ₹{rmse:.1f} Lakhs")
print(f"✅ R²: {r2:.3f}")
