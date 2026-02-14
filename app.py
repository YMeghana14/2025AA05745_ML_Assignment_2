import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.title("ML Assignment 2 - Classification Models")
st.write("Adult Income Dataset - Model Comparison")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","income"
    ]
    df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
    df.dropna(inplace=True)
    return df

df = load_data()

# ----------------------------
# Preprocessing
# ----------------------------
df = pd.get_dummies(df, drop_first=True)

X = df.drop("income_>50K", axis=1)
y = df["income_>50K"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# Model Selection
# ----------------------------
st.sidebar.header("Select Model")

model_name = st.sidebar.selectbox(
    "Choose a model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)

elif model_name == "Decision Tree":
    model = DecisionTreeClassifier()

elif model_name == "KNN":
    model = KNeighborsClassifier()

elif model_name == "Naive Bayes":
    model = GaussianNB()

elif model_name == "Random Forest":
    model = RandomForestClassifier()

elif model_name == "XGBoost":
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ----------------------------
# Evaluation Metrics
# ----------------------------
st.subheader("Evaluation Metrics")

st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))
st.write("F1 Score:", f1_score(y_test, y_pred))
st.write("AUC:", roc_auc_score(y_test, y_prob))
st.write("MCC:", matthews_corrcoef(y_test, y_pred))

# ----------------------------
# Confusion Matrix
# ----------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ----------------------------
# Classification Report
# ----------------------------
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))
