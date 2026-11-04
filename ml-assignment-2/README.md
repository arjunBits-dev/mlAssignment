# ML Assignment 2 — Classification Model Comparison

> M.Tech (AIML), BITS Pilani WILP — Machine Learning Assignment 2

## a. Problem Statement

<!-- TODO: 2-4 lines describing the classification problem your dataset addresses. -->

## b. Dataset Description

<!-- TODO: Fill in after dataset selection.
- Source: (Kaggle/UCI link)
- Number of instances: (must be >= 500)
- Number of features: (must be >= 12)
- Target variable and classes (binary / multi-class)
- Brief description of key features
-->

## c. GitHub Repository Link

<!-- TODO: paste repo URL here -->

## d. Models Used

The following five classifiers were trained on the same dataset:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

### Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| SVM | | | | | | |

### Observations

| Model | Observation on performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| SVM | |

**Overall winner for this dataset:** <!-- TODO: name the model and justify why -->

## Streamlit App

Live app: <!-- TODO: paste Streamlit Community Cloud URL -->

### Features

- Upload test data (CSV with all input features + target column)
- Select any of the five trained models from a dropdown
- View evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- View confusion matrix / classification report

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository Structure

```
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── test_data.csv       # Sample test data (features + target)
├── model/              # Saved trained models + preprocessing artifacts
└── notebooks/          # Training / EDA source code
```
