import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("student_exam_data.csv")
print(df.head())
X = df[['Study Hours']]
y = df['Pass/Fail']

model = LogisticRegression()
model.fit(X, y)

prob_pass = model.predict_proba([[6]])[0][1]
print("Probability of passing for 6 study hours:", prob_pass)

X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_prob = model.predict_proba(X_range)[:, 1]

plt.scatter(X, y)
plt.plot(X_range, y_prob)
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression Sigmoid Curve")
plt.show()
