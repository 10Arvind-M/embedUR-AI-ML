import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------------
# Load MNIST Dataset
# -----------------------------------

print("Loading MNIST Dataset...")

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

print("Dataset Loaded Successfully")

# -----------------------------------
# Normalize Pixel Values
# -----------------------------------

X = X / 255.0

# -----------------------------------
# Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# -----------------------------------
# Train Logistic Regression Model
# -----------------------------------

model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs',
    multi_class='auto'
)

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Training Completed")

# -----------------------------------
# Predictions
# -----------------------------------

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# -----------------------------------
# Accuracy
# -----------------------------------

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("\nTraining Accuracy :", round(train_accuracy*100,2),"%")
print("Testing Accuracy  :", round(test_accuracy*100,2),"%")

# -----------------------------------
# Save Results
# -----------------------------------

with open("results.txt","w") as file:

    file.write("MNIST DIGIT CLASSIFIER\n")
    file.write("========================\n\n")
    file.write(f"Training Accuracy : {train_accuracy*100:.2f}%\n")
    file.write(f"Testing Accuracy  : {test_accuracy*100:.2f}%\n")

# -----------------------------------
# Correct Predictions
# -----------------------------------

correct = np.where(test_pred == y_test)[0][:5]

plt.figure(figsize=(12,3))

for i,index in enumerate(correct):

    plt.subplot(1,5,i+1)

    plt.imshow(X_test[index].reshape(28,28), cmap="gray")

    plt.title(f"P:{test_pred[index]}\nA:{y_test[index]}")

    plt.axis("off")

plt.suptitle("Correct Predictions")

plt.savefig("correct_predictions.png")

plt.show()

# -----------------------------------
# Incorrect Predictions
# -----------------------------------

incorrect = np.where(test_pred != y_test)[0][:5]

plt.figure(figsize=(12,3))

for i,index in enumerate(incorrect):

    plt.subplot(1,5,i+1)

    plt.imshow(X_test[index].reshape(28,28), cmap="gray")

    plt.title(f"P:{test_pred[index]}\nA:{y_test[index]}")

    plt.axis("off")

plt.suptitle("Incorrect Predictions")

plt.savefig("incorrect_predictions.png")

plt.show()
