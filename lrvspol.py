import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("Bengaluru_House_Data.csv")
print(df.head())

df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')
df = df.dropna(subset=['total_sqft', 'price'])

X = df[['total_sqft']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred_lin = lin.predict(X_test)

print("Linear Coef:", lin.coef_[0])
print("Linear Intercept:", lin.intercept_)
print("Linear R2:", r2_score(y_test, y_pred_lin))


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(X_poly, y, test_size=0.2, random_state=42)

poly_model = LinearRegression()
poly_model.fit(X_tr, y_tr)
y_pred_poly = poly_model.predict(X_te)

print("Polynomial R2:", r2_score(y_te, y_pred_poly))

print("\nPolynomial Coefficients:", poly_model.coef_)
print("Polynomial Intercept:", poly_model.intercept_)

X_range = np.linspace(X.min(), X.max(), 100)

plt.scatter(X, y)

# Linear line
plt.plot(X_range, lin.predict(X_range), label="Linear Regression")

# Polynomial curve
X_range_poly = poly.transform(X_range)
plt.plot(X_range, poly_model.predict(X_range_poly), label="Polynomial Regression")

plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
