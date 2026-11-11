"""Task 4: Credit Card Fraud Detection (imbalanced data)
Dataset: Kaggle 'Credit Card Fraud Detection' -> save as creditcard.csv (target column 'Class')
Run: python task4_credit_card_fraud_detection.py
"""
import os
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

if os.path.exists("creditcard.csv"):
    df = pd.read_csv("creditcard.csv")
    df["Amount"] = StandardScaler().fit_transform(df[["Amount"]])
    df = df.drop(columns=["Time"], errors="ignore")
    X, y = df.drop(columns="Class"), df["Class"]
else:
    print("creditcard.csv not found -> using synthetic imbalanced data")
    X, y = make_classification(n_samples=20000, n_features=20, weights=[0.99, 0.01],
                               n_informative=8, random_state=42)

print("Class distribution:\n", pd.Series(y).value_counts(normalize=True))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Handling imbalance: class weights (+ undersampling the majority for Random Forest)
models = {
    "Logistic Regression (class_weight=balanced)":
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest (balanced_subsample)":
        RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                               n_jobs=-1, random_state=42),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(f"Precision: {precision_score(y_test, pred):.4f}  "
          f"Recall: {recall_score(y_test, pred):.4f}  F1: {f1_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, target_names=["legit", "fraud"]))
