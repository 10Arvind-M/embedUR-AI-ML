# AI for Everyone – Coding, Data Processing & Basic ML Assignments

## Overview

This repository contains solutions for all ten assignments covering Python programming, data processing, machine learning, and deep learning. The assignments progress from basic data analysis to image classification and transfer learning using TensorFlow.

---

## Environment

- Python 3.11+
- Operating System: Windows/Linux/macOS

---

## Required Libraries

### Core Libraries

- pandas
- numpy
- matplotlib
- scikit-learn

### Deep Learning Libraries (Questions 8 & 9)

- tensorflow
- opencv-python

## Installation

Install the required libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow opencv-python
```

# Project Structure

```
q1/
q2/
q3/
q4/
q5/
q6/
q7/
q8/
q9/
q10/
```

Each folder contains:

- Dataset
- Python source code
- Output files
- Generated plots
- Result files

---
---

# Datasets

The datasets used for each assignment can be downloaded from the following sources:

| Question | Dataset | Download Link |
|----------|---------|---------------|
| Q1 | Sales CSV Dataset | https://www.kaggle.com/datasets/anairamcosta/sales-csv |
| Q2 | Sample Application Log File | Included in this repository (`application.log`) |
| Q3 | Iris Dataset | https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html |
| Q4 | MNIST Handwritten Digits | https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html *(or TensorFlow MNIST dataset if used)* |
| Q5 | Titanic - Machine Learning from Disaster | https://www.kaggle.com/competitions/titanic/data |
| Q6 | SMS Spam Collection Dataset | https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset |
| Q7 | Rock Paper Scissors Dataset | https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors |
| Q8 | Casting Product Image Dataset | https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product |
| Q9 | Face Mask Detection Dataset | https://www.kaggle.com/datasets/omkargurav/face-mask-dataset |
| Q10 | Heart Disease Dataset | https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset |

> **Note:** Large image datasets (Questions 7, 8 and 9) are not included in this repository due to GitHub file size limitations. Download the datasets from the above links and place them inside their respective assignment folders before running the code.

---
# Assignment Summary

## Question 1 – Customer Sales Analysis

- Load CSV using Pandas
- Handle missing values
- Compute revenue by product category
- Identify top customers (or equivalent analysis depending on dataset)
- Visualize monthly sales

Output:

- Cleaned Dataset
- Summary Statistics
- Revenue Plot
- Monthly Sales Plot

---

## Question 2 – Log File Analyzer

- Parse application log file
- Identify ERROR and WARNING entries
- Group by module
- Generate CSV report

Output:

- Error Report CSV

---

## Question 3 – Iris Classification

- Load Iris Dataset
- Train Decision Tree Classifier
- Evaluate Accuracy
- Generate Confusion Matrix

Output:

- Confusion Matrix
- Evaluation Metrics

---

## Question 4 – MNIST Digit Recognition

- Load MNIST Dataset
- Train Logistic Regression
- Evaluate Training/Test Accuracy
- Display Correct and Incorrect Predictions

Output:

- Training Accuracy
- Test Accuracy
- Prediction Images

---

## Question 5 – Titanic Survival Prediction

- Exploratory Data Analysis
- Missing Value Handling
- Feature Engineering
- Logistic Regression
- Random Forest
- Model Comparison

Output:

- Evaluation Metrics
- EDA Plots

---

## Question 6 – Spam Email Classification

- TF-IDF Feature Extraction
- Naive Bayes Classifier
- Precision
- Recall
- F1-Score

Output:

- Classification Report

---

## Question 7 – Image Classification Pipeline

- Image Loading
- Image Resizing
- Pixel Normalization
- Train/Validation Split
- Metadata Generation

Output:

- Metadata CSV
- Training CSV
- Validation CSV

---

## Question 8 – Manufacturing Defect Detection

- Image Preprocessing
- Data Augmentation
- CNN Training
- Model Evaluation
- Deployment Considerations

Output:

- Trained Model
- Accuracy Plot
- Loss Plot

---

## Question 9 – Face Mask Detection

- MobileNetV2 Transfer Learning
- Fine-Tuning
- Precision
- Recall
- F1-Score
- Inference Speed

Output:

- Trained Model
- Accuracy Plot
- Loss Plot

---

## Question 10 – End-to-End Machine Learning Project

Dataset Used:

Heart Disease Prediction Dataset

Workflow:

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Model Training
- Model Evaluation
- Error Analysis
- Business Insights

Output:

- Evaluation Metrics
- Correlation Heatmap
- Target Distribution Plot

---

# How to Run

Navigate to the required assignment folder.

Example:

```bash
cd q3
```

Run the Python script:

```bash
python answer.py
```

---

# Outputs

Depending on the assignment, the following files are generated:

- Cleaned datasets
- CSV reports
- PNG plots
- Confusion matrices
- Evaluation metrics
- Trained models (Deep Learning assignments)

# Technologies Used

- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow
- OpenCV

---

# Conclusion

These assignments demonstrate the complete machine learning workflow, beginning with data preprocessing and visualization, progressing through classical machine learning algorithms, and concluding with deep learning and transfer learning techniques. The solutions follow standard machine learning practices and emphasize data preparation, model evaluation, and interpretation of results.
