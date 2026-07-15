import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------------
# Load Dataset
# -----------------------------------------

df = pd.read_csv("heart.csv")

print("First 5 Records")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# -----------------------------------------
# Data Cleaning
# -----------------------------------------

df.drop_duplicates(inplace=True)

# -----------------------------------------
# Exploratory Data Analysis
# -----------------------------------------

plt.figure(figsize=(5,4))

df["target"].value_counts().plot(kind="bar")

plt.title("Heart Disease Distribution")

plt.xlabel("Target")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig("target_distribution.png")

plt.show()

# -----------------------------------------
# Correlation Heatmap
# -----------------------------------------

plt.figure(figsize=(10,8))

plt.imshow(df.corr(), cmap="coolwarm")

plt.colorbar()

plt.xticks(range(len(df.columns)), df.columns, rotation=90)

plt.yticks(range(len(df.columns)), df.columns)

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.show()

# -----------------------------------------
# Features
# -----------------------------------------

X = df.drop("target", axis=1)

y = df["target"]

# Age Groups
df["AgeGroup"] = pd.cut(
    df["age"],
    bins=[20,40,50,60,80],
    labels=[0,1,2,3]
).astype(int)

# Cholesterol per Age
df["CholAge"] = df["chol"] / df["age"]

# Heart Rate per Age
df["HR_Age"] = df["thalach"] / df["age"]

# High Blood Pressure
df["HighBP"] = (df["trestbps"] >= 140).astype(int)

# -----------------------------------------
# Train Test Split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    stratify=y,

    random_state=7
)

# -----------------------------------------
# Random Forest
# -----------------------------------------

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV

params = {

    "n_estimators":[200,300,500],

    "max_depth":[4,6,8,None],

    "min_samples_split":[2,5,10],

    "min_samples_leaf":[1,2,4]

}

search = RandomizedSearchCV(

    ExtraTreesClassifier(random_state=42),

    params,

    cv=5,

    scoring="accuracy",

    n_iter=20,

    random_state=42

)

search.fit(X_train,y_train)

model = search.best_estimator_
# -----------------------------------------
# Prediction
# -----------------------------------------

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\nAccuracy :", accuracy)

print("\nClassification Report")

print(classification_report(y_test, pred))

print("\nConfusion Matrix")

print(confusion_matrix(y_test, pred))

# -----------------------------------------
# Feature Importance
# -----------------------------------------

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nFeature Importance")

print(importance)

# -----------------------------------------
# Save Model
# -----------------------------------------

joblib.dump(

    model,

    "heart_model.pkl"

)

# -----------------------------------------
# Save Results
# -----------------------------------------

with open("results.txt", "w") as file:

    file.write("Heart Disease Prediction\n\n")

    file.write(f"Accuracy : {accuracy:.4f}\n\n")

    file.write(classification_report(y_test, pred))

    file.write("\n\nFeature Importance\n")

    file.write(importance.to_string(index=False))

