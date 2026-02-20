import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("car_price_prediction.csv")

print("Dataset Preview:")
print(df.head())
# 3️⃣ Feature Engineering: Calculate Age
df['Age'] = 2026 - df['Prod. year']  # assuming current year 2026

# Select Features and Target
X = df[['Mileage', 'Age', 'Engine volume']]  # corrected column name
y = df['Price']

# Handle missing values if any
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# 4️⃣ Split Data into Training and Testing Sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Train Multiple Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6️⃣ Predict on Test Set
y_pred = model.predict(X_test)

# 7️⃣ Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error on Test Set:", mse)

# 8️⃣ Coefficients and Intercept
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nFeature Coefficients:")
print(coefficients)

intercept = model.intercept_
print("\nIntercept:", intercept)

# 9️⃣ Regression Equation
print(f"\nRegression Equation:")
print(f"Price = {intercept:.2f} + ({coefficients['Coefficient'][0]:.2f} * Mileage) + "
      f"({coefficients['Coefficient'][1]:.2f} * Age) + "
      f"({coefficients['Coefficient'][2]:.2f} * Engine volume)")

# 10️⃣ Interpretation
print("\nInterpretation of Coefficients:")
for i, row in coefficients.iterrows():
    feature = row['Feature']
    coef = row['Coefficient']
    if coef > 0:
        effect = "increases"
    else:
        effect = "decreases"
    print(f"- A 1 unit increase in {feature} {effect} the Price by {abs(coef):.2f} units, holding other features constant.")