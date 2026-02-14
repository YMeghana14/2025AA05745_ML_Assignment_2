import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report

st.title("ML Assignment 2 - Classification Models Demo_2025AA05745")

st.write("Upload a CSV file (test dataset) containing the same features as the training data.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "income" not in df.columns:
        st.error("Uploaded file must contain 'income' column.")
    else:
        X = df.drop("income", axis=1)
        y = df["income"]

        model = pickle.load(open(f"{model_choice}.pkl", "rb"))
        y_pred = model.predict(X)

        st.subheader("Evaluation Metrics")

        st.write("Accuracy:", accuracy_score(y, y_pred))
        st.write("Precision:", precision_score(y, y_pred))
        st.write("Recall:", recall_score(y, y_pred))
        st.write("F1 Score:", f1_score(y, y_pred))
        st.write("AUC:", roc_auc_score(y, y_pred))
        st.write("MCC:", matthews_corrcoef(y, y_pred))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))
