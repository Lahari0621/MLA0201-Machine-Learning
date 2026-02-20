# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
data = pd.read_csv("Bengaluru_House_Data.csv")

# Select required columns
data = data[['total_sqft', 'bedroom', 'price']].dropna()

# Convert total_sqft to numeric
data['total_sqft'] = pd.to_numeric(data['total_sqft'], errors='coerce')
data = data.dropna()

# Convert price into categories (classification)
def price_category(price):
    if price < 50:
        return 0      # Low
    elif price < 100:
        return 1      # Medium
    else:
        return 2      # High

data['price_class'] = data['price'].apply(price_category)

# Features and target
X = data[['total_sqft', 'bedroom']]
y = data['price_class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Naïve Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Output results
print("Naïve Bayes Classification Results")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))
