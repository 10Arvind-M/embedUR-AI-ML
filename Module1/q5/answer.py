import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# ----------------------------------------
# Load Dataset
# ----------------------------------------

df = pd.read_csv("dataset.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Info")
print(df.info())

# ----------------------------------------
# Exploratory Data Analysis (EDA)
# ----------------------------------------

print("\nMissing Values")
print(df.isnull().sum())

print("\nSummary Statistics")
print(df.describe())

# ----------------------------------------
# Plot 1 : Survival Distribution
# ----------------------------------------

plt.figure(figsize=(5,4))
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("survival_distribution.png")
plt.show()

# ----------------------------------------
# Plot 2 : Age Distribution
# ----------------------------------------

plt.figure(figsize=(6,4))
df["Age"].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.show()

# ----------------------------------------
# Data Cleaning
# ----------------------------------------

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df.drop(columns=["Cabin"], inplace=True)

# ----------------------------------------
# Feature Engineering
# ----------------------------------------

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# ----------------------------------------
# Encode Categorical Columns
# ----------------------------------------

encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])

# ----------------------------------------
# Select Features
# ----------------------------------------

features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "FamilySize"
]

X = df[features]

y = df["Survived"]

# ----------------------------------------
# Split Dataset
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------
# Logistic Regression
# ----------------------------------------

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Accuracy")
print(lr_accuracy)

# ----------------------------------------
# Random Forest
# ----------------------------------------

rf = RandomForestClassifier(
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy")
print(rf_accuracy)

# ----------------------------------------
# Classification Reports
# ----------------------------------------

print("\nLogistic Regression Report")
print(classification_report(y_test, lr_pred))

print("\nRandom Forest Report")
print(classification_report(y_test, rf_pred))

# ----------------------------------------
# Save Results
# ----------------------------------------

with open("results.txt", "w") as file:

    file.write("Titanic Survival Prediction\n")
    file.write("=============================\n\n")

    file.write(f"Logistic Regression Accuracy : {lr_accuracy:.4f}\n")
    file.write(f"Random Forest Accuracy       : {rf_accuracy:.4f}\n\n")

    if rf_accuracy > lr_accuracy:
        file.write("Recommended Model : Random Forest\n")
    else:
        file.write("Recommended Model : Logistic Regression\n")

