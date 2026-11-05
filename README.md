# ML Assignment 2 — Online Shoppers Purchasing Intention

Machine Learning Assignment 2 — M.Tech (AIML), BITS Pilani WILP.

This repository contains my submission for ML Assignment 2. Five
classification models are trained on the same dataset and compared using six
evaluation metrics. A Streamlit app loads the saved models and evaluates them
on uploaded test data (no retraining happens in the app).

---

## a. Problem Statement

The goal is to predict whether a visitor browsing an online shopping website
will end up making a purchase in that session or not. The dataset has
information about each session, like the types of pages visited, time spent
on them, bounce and exit rates, and so on. If the website can predict the
purchase intention while the user is still browsing, it can react in time,
for example by showing a discount coupon or a chat popup to a visitor who
looks interested but is about to leave.

This is a **binary classification** problem. The target column is `Revenue`
(`True` = the session ended with a purchase, `False` = it did not) and there
are 17 input features.

## b. Dataset Description

- **Source:** [UCI Machine Learning Repository — Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Instances:** 12,330 browsing sessions (each row = one user session over a 1-year period)
- **Features:** 17 input features + 1 target column
- **Target:** `Revenue` — binary (True/False). **Imbalanced:** only 15.47% of
  sessions end in a purchase, which makes metrics such as AUC, F1 and MCC far
  more informative than plain accuracy.
- **Missing values:** none.

### Feature overview

| Group | Features | Notes |
|---|---|---|
| Page counts & durations (numeric) | `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration` | How many pages of each type were visited and for how long |
| Behaviour metrics (numeric) | `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay` | Google-Analytics style session metrics; `PageValues` is the strongest purchase signal |
| Session context (categorical) | `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType` | `OperatingSystems`/`Browser`/`Region`/`TrafficType` are integer-coded category IDs |
| Binary | `Weekend` | Whether the session happened on a weekend |

### Preprocessing

- Stratified **80/20 train/test split** (`random_state=42`) — both splits keep
  the 15.5% purchase rate.
- **StandardScaler** on the 10 numeric features (required for KNN, SVM and
  helps Logistic Regression converge).
- **One-hot encoding** on the 6 categorical features (17 raw features expand
  to 74 model inputs).
- The fitted preprocessing pipeline is saved to `model/preprocessor.joblib`
  and reused by the Streamlit app — uploaded raw CSVs go through the exact
  same transformation as the training data.

## c. GitHub Repository Link

> <https://github.com/arjunBits-dev/mlAssignment>

## d. Models Used

All five classifiers were trained on the same preprocessed training split and
evaluated on the same unseen test split (2,466 sessions):

1. Logistic Regression (`max_iter=1000`)
2. Decision Tree
3. K-Nearest Neighbors (`n_neighbors=11`)
4. Naive Bayes (GaussianNB)
5. Support Vector Machine (RBF kernel, `probability=True`)

### Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8812 | **0.8830** | **0.7405** | 0.3586 | 0.4832 | 0.4610 |
| Decision Tree | 0.8593 | 0.7340 | 0.5452 | 0.5524 | 0.5488 | 0.4654 |
| KNN | 0.8743 | 0.8186 | 0.6957 | 0.3351 | 0.4523 | 0.4244 |
| Naive Bayes | 0.2733 | 0.7330 | 0.1727 | **0.9738** | 0.2934 | 0.1292 |
| **SVM** | **0.8873** | 0.8817 | 0.7063 | 0.4660 | **0.5615** | **0.5142** |

### Observations

| Model | Observation on performance |
|---|---|
| Logistic Regression | Gave the best AUC (0.8830) and the highest precision (0.7405) among all five models. But recall is low (~36%), so it misses many of the actual buyers. It ranks sessions well but is too conservative when predicting the purchase class. |
| Decision Tree | The only model where precision and recall came out almost equal (~0.55 each). However its AUC is the lowest of the useful models (0.7340). The tree was grown fully without pruning, so it most likely overfits the training data, which also explains the weak probability estimates. |
| KNN | Accuracy looks okay (0.8743) but recall is only 33.5%. After one-hot encoding the data has 74 columns, and KNN struggles in high dimensions because distances between points start looking similar for everyone. |
| Naive Bayes | Behaved completely differently from the rest. Recall is 97.4% but accuracy dropped to just 27.3%, because it predicts "purchase" for almost every session. GaussianNB assumes the features are independent and normally distributed, and our one-hot columns and skewed duration features break both assumptions. The MCC of 0.1292 shows the predictions are close to useless. |
| SVM | The best scores in accuracy (0.8873), F1 (0.5615) and MCC (0.5142), with AUC almost equal to logistic regression. The RBF kernel can pick up non-linear patterns in the data that the linear models missed, and feature scaling kept all inputs comparable. |

**Overall winner for this dataset: SVM.**
Since the dataset is imbalanced (only ~15% of sessions are purchases), I did
not judge the models on accuracy alone — even predicting "no purchase" every
single time would already give about 84.5% accuracy. Going by F1 and MCC,
which handle imbalance much better, SVM is clearly the strongest model here.
It catches around 46% of the real buyers, and when it does predict a purchase
it is right about 70% of the time, which is the best trade-off among the five
models.

---

## Streamlit App

**Live app:** <https://mlassignment2-2026aj05425.streamlit.app>

### Features

- **Upload test data** — CSV containing all 17 input features + the `Revenue` target column (the repo's `test_data.csv` is a ready-made example, 2,466 rows)
- **Model selection dropdown** — choose any of the five trained classifiers
- **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1 and MCC computed on the uploaded data
- **Confusion matrix + classification report** for the selected model

The app **loads the saved models and preprocessor** from the `model/` folder
and never retrains on uploaded data.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload `test_data.csv` in the sidebar.

## Repository Structure

```
├── app.py                  # Streamlit application
├── requirements.txt        # Pinned Python dependencies
├── README.md               # This file
├── test_data.csv           # Test split (17 features + Revenue), for the app
├── data/
│   └── online_shoppers_intention.csv   # Original UCI dataset
├── model/
│   ├── preprocessor.joblib             # Fitted scaling + encoding pipeline
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
└── notebooks/
    └── 2026aj05425_ML_Assignment2.ipynb  # EDA, preprocessing, training, evaluation
```

## Reproducibility

- Fixed random seed (`random_state=42`) for the split and all models.
- `requirements.txt` pins the exact library versions used for training, so
  the deployed app loads the saved models with the same scikit-learn version.
- The notebook's final section reloads every saved model from disk and
  verifies it reproduces the reported metrics exactly.
