import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt  

df = pd.read_csv("student_exam_scores.csv")
print(df.head())
x = df[['hours_studied']]
y = df['exam_score']
plt.scatter(x, y)
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.show()
model = LinearRegression()
model.fit(x, y)
prediction = model.predict([[8]])
print("Predicted score for 8 study hours:", prediction[0])
plt.scatter(x, y)
plt.plot(x, model.predict(x))
plt.xlabel("Study Hours")
plt.ylabel("Previous Exam Score")
plt.title("Linear Regression Line")
plt.show()

