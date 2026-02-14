# ML Assignment 2 - Classification Models

## Problem Statement
The objective of this project is to implement multiple machine learning classification models on the Adult Income dataset and compare their performance using various evaluation metrics.

## Dataset Description
The Adult Income dataset is a binary classification dataset used to predict whether a person's income exceeds 50K per year.  
It contains more than 48,000 instances and 14 features including age, education, occupation, work hours, etc.

## Models Implemented
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Naive Bayes
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## Evaluation Metrics Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.8272 | 0.7062 | 0.7230 | 0.4705 | 0.5700 | 0.4848 |
| Decision Tree | 0.8155 | 0.7575 | 0.6156 | 0.6444 | 0.6297 | 0.5071 |
| KNN | 0.8318 | 0.7575 | 0.6688 | 0.6126 | 0.6395 | 0.5310 |
| Naive Bayes | 0.8014 | 0.6467 | 0.6822 | 0.3451 | 0.4584 | 0.3832 |
| Random Forest | 0.8620 | 0.7887 | 0.7524 | 0.6458 | 0.6950 | 0.6095 |
| XGBoost | 0.8755 | 0.8105 | 0.7779 | 0.6839 | 0.7279 | 0.6499 |

## Observations

- Logistic Regression performs moderately well but struggles with recall.
- Decision Tree improves recall but slightly overfits.
- KNN provides balanced performance.
- Naive Bayes performs weakest due to independence assumption.
- Random Forest significantly improves performance using ensemble learning.
- XGBoost gives the best overall performance with highest accuracy and MCC.

## Deployment
The application is deployed using Streamlit Community Cloud.
