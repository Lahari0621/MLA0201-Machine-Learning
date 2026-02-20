import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("Screen time of kids in india.csv")
print(df.head)
y = df['Avg_Daily_Screen_Time_hr']
X = df.drop('Avg_Daily_Screen_Time_hr', axis=1)
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_lr = X_train[['Age']]
X_test_lr = X_test[['Age']]
linear_model = LinearRegression()
linear_model.fit(X_train_lr, y_train)
y_pred_lr = linear_model.predict(X_test_lr)
mse_linear = mean_squared_error(y_test, y_pred_lr)
print("\nLinear Regression MSE:", mse_linear)
multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)
y_pred_mlr = multiple_model.predict(X_test)
mse_multiple = mean_squared_error(y_test, y_pred_mlr)
print("Multiple Linear Regression MSE:", mse_multiple)
print("\n--- FINAL MSE COMPARISON ---")
print("Linear Regression MSE   :", mse_linear)
print("Multiple Regression MSE :", mse_multiple)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_lr, label='Linear Regression Predictions', alpha=0.6)
sns.scatterplot(x=y_test, y=y_pred_mlr, label='Multiple Linear Regression Predictions', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
plt.title('Actual vs. Predicted Average Daily Screen Time (Test Set)')
plt.xlabel('Actual Avg Daily Screen Time (hr)')
plt.ylabel('Predicted Avg Daily Screen Time (hr)')
plt.legend()
plt.grid(True)
plt.show()