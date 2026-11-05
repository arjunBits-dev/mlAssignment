# ML Assignment 2 — Online Shoppers Purchasing Intention

Machine Learning Assignment 2 — M.Tech (AIML), BITS Pilani WILP.

Five classification models are trained on the same dataset, compared with six
evaluation metrics, and served through a Streamlit web app that evaluates the
saved models on uploaded test data (without retraining).

---

## a. Problem Statement

An e-commerce website wants to know, **while a visitor is still browsing**,
whether the session is likely to end in a purchase. If purchase intent can be
predicted from the visitor's browsing behaviour (pages visited, time spent,
bounce/exit rates, etc.), the site can act in real time — for example by
showing an offer, a discount coupon, or live chat support to visitors who are
close to buying but may leave.

This is a **binary classification** problem: predict the target `Revenue`
(`True` = session ended in a purchase, `False` = it did not) from 17 session
features.

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
| Logistic Regression | Best AUC (0.8830) and highest precision — when it predicts "purchase" it is usually right. However it is conservative: recall is only ~36%, so it misses most actual buyers. A good *ranking* model whose threshold could be tuned. |
| Decision Tree | The most balanced precision/recall of the five, but the lowest AUC among the usable models (0.7340) — a single unpruned tree overfits the training data and produces poor probability estimates. |
| KNN | Reasonable accuracy but weak recall (33.5%). After one-hot encoding the data has 74 dimensions, where distance between points becomes less meaningful (curse of dimensionality), limiting KNN. |
| Naive Bayes | Extreme behaviour: 97.4% recall but only 27.3% accuracy — it predicts "purchase" for almost every session. GaussianNB assumes independent, normally-distributed features; one-hot columns and highly skewed duration features violate both assumptions badly. Lowest MCC (0.1292) confirms predictions are close to uninformative. |
| SVM | Best overall: highest accuracy (0.8873), F1 (0.5615) and MCC (0.5142), with AUC essentially tied with Logistic Regression. The RBF kernel captures non-linear feature interactions while scaling keeps all features comparable. |

**Overall winner for this dataset: SVM.**
On an imbalanced dataset, accuracy alone is misleading (always predicting
"no purchase" would already score 84.5%). Judged by the metrics that account
for imbalance — MCC and F1 — the SVM is the clear winner, while also having
the top accuracy and near-best AUC. It offers the best balance between
catching real buyers (46.6% recall) and not crying wolf (70.6% precision).

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
