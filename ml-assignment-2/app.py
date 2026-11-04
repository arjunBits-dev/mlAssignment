# Streamlit app — ML Assignment 2
# Loads SAVED trained models from model/ and evaluates them on uploaded test data.
# NOTE: models are never retrained here.
#
# Planned features (per assignment requirements):
#   1. CSV test-data upload (features + target column)
#   2. Model selection dropdown (LogReg, Decision Tree, KNN, Naive Bayes, SVM)
#   3. Evaluation metrics display (Accuracy, AUC, Precision, Recall, F1, MCC)
#   4. Confusion matrix / classification report

import streamlit as st

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Classification Model Comparison")
st.info("Skeleton placeholder — implementation pending.")
