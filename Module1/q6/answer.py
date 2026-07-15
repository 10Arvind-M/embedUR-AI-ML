import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,classification_report


# ---------------------------------------
# Load Dataset
# ---------------------------------------

# ---------------------------------------
# Load Dataset
# ---------------------------------------

df = pd.read_csv(
    "dataset.csv",
    sep="\t",
    header=None,
    names=["Label", "Message"]
)

print("First 5 Records")
print(df.head())

print("\nDataset Information")
print(df.info())

# Rename columns
df.columns = ["Label", "Message"]

print("First 5 Records")
print(df.head())

print("\nDataset Information")
print(df.info())

# ---------------------------------------
# Encode Labels
# ham = 0
# spam = 1
# ---------------------------------------

df["Label"] = df["Label"].map({
    "ham": 0,
    "spam": 1
})

# ---------------------------------------
# Train-Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["Message"],
    df["Label"],
    test_size=0.2,
    random_state=42
)

# ---------------------------------------
# TF-IDF Feature Extraction
# ---------------------------------------

vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------
# Train Naive Bayes Model
# ---------------------------------------

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

# ---------------------------------------
# Predictions
# ---------------------------------------

y_pred = model.predict(X_test_tfidf)

# ---------------------------------------
# Evaluation Metrics
# ---------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy,4))
print("Precision:", round(precision,4))
print("Recall   :", round(recall,4))
print("F1 Score :", round(f1,4))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# ---------------------------------------
# Save Results
# ---------------------------------------

with open("results.txt", "w") as file:

    file.write("Spam Email Classification\n")
    file.write("===========================\n\n")

    file.write(f"Accuracy  : {accuracy:.4f}\n")
    file.write(f"Precision : {precision:.4f}\n")
    file.write(f"Recall    : {recall:.4f}\n")
    file.write(f"F1 Score  : {f1:.4f}\n\n")

    file.write("Classification Report\n")
    file.write(classification_report(y_test, y_pred))

print("\nResults saved successfully.")
