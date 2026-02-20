import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   
import matplotlib.pyplot as plt

df=pd.read_csv("Salary_Data.csv")
print(df.head())    
x=df[['YearsExperience']]
y=df['Salary'] 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(x_train, y_train)

salary_5yrs = model.predict([[5]])
print("Predicted Salary for 5 years experience:", salary_5yrs[0])

plt.scatter(x, y)
plt.plot(x, model.predict(x))
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Years of Experience vs Salary")
plt.show()

y_pred = model.predict(x_test)
residuals = y_test - y_pred

plt.scatter(x_test, residuals)
plt.axhline(y=0)
plt.xlabel("Years of Experience")
plt.ylabel("Residual Error")
plt.title("Residual Errors Plot")
plt.show()
