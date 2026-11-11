"""Task 1: Spam SMS Detection (TF-IDF + Naive Bayes / Logistic Regression)
Dataset: SMS Spam Collection (Kaggle/UCI) -> save as spam.csv (columns v1=label, v2=text)
Run: python task1_spam_sms_detection.py
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

if os.path.exists("spam.csv"):
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
else:  # tiny demo data so the script always runs
    print("spam.csv not found -> using tiny demo data")
    df = pd.DataFrame({
        "label": ["spam", "ham"] * 20,
        "text": ["WIN a free prize now, click the link!", "Are we meeting for lunch today?"] * 20})

df["y"] = (df["label"] == "spam").astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"])

tfidf = TfidfVectorizer(stop_words="english", lowercase=True)
Xtr, Xte = tfidf.fit_transform(X_train), tfidf.transform(X_test)

models = {"Naive Bayes": MultinomialNB(),
          "Logistic Regression": LogisticRegression(max_iter=1000)}
for name, model in models.items():
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=["ham", "spam"]))

# Try your own messages
best = models["Logistic Regression"]
for msg in ["Congratulations! You won a free ticket, call now", "Hey, see you at 6?"]:
    p = best.predict(tfidf.transform([msg]))[0]
    print(f"'{msg}' -> {'SPAM' if p else 'LEGITIMATE'}")
