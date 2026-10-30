# Iris Flower Classification

# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create Machine Learning Model
model = DecisionTreeClassifier(random_state=42)

# Step 5: Train Model
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Check Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Iris Flower Classification")
print("---------------------------")
print("Total Dataset:", len(X))
print("Training Data:", len(X_train))
print("Testing Data:", len(X_test))
print("Model Accuracy:", accuracy * 100, "%")

# Step 8: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=iris.target_names))

# Step 9: Predict New Flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Predicted Flower:", iris.target_names[prediction[0]])
# Step 10: Visualize Iris Dataset

plt.figure(figsize=(8, 5))

for i in range(3):
    plt.scatter(
        X[y == i, 0],
        X[y == i, 2],
        label=iris.target_names[i]
    )

plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")

plt.title("Iris Flower Classification")

plt.legend()
plt.grid()
plt.show()
from sklearn.metrics import ConfusionMatrixDisplay

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=iris.target_names
)

plt.title("Iris Classification Confusion Matrix")
plt.show()