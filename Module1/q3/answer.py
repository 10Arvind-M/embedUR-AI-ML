# Import required libraries

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# --------------------------------------
# Load Iris Dataset
# --------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Loaded Successfully")
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)

# --------------------------------------
# Split Dataset
# --------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# --------------------------------------
# Train Decision Tree Classifier
# --------------------------------------

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# --------------------------------------
# Predictions
# --------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------
# Accuracy
# --------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

# --------------------------------------
# Classification Report
# --------------------------------------

report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
)

print("\nClassification Report")
print(report)

# --------------------------------------
# Confusion Matrix
# --------------------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()

# --------------------------------------
# Save Results
# --------------------------------------

with open("results.txt", "w") as file:

    file.write("IRIS CLASSIFICATION RESULTS\n")
    file.write("=============================\n\n")

    file.write(f"Accuracy : {accuracy*100:.2f}%\n\n")

    file.write("Classification Report\n")
    file.write(report)

print("\nResults saved successfully.")