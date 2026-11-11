"""Task 2: Sentiment Analysis (positive / negative / neutral)
Dataset: any reviews/tweets CSV with columns 'text' and 'sentiment'
(e.g. Kaggle 'Twitter Entity Sentiment Analysis' / 'Amazon Reviews') -> save as reviews.csv
Run: python task2_sentiment_analysis.py
"""
import os, re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#", " ", text)       # urls, mentions, hashtags
    text = re.sub(r"[^a-z\s]", " ", text)              # punctuation & numbers
    return " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

if os.path.exists("reviews.csv"):
    df = pd.read_csv("reviews.csv")[["text", "sentiment"]].dropna()
else:
    print("reviews.csv not found -> using tiny demo data")
    pos = ["I love this product, it is great!", "Amazing quality and fast delivery", "Really happy with it"]
    neg = ["Terrible, it broke in a day", "Worst purchase ever, very disappointed", "Awful quality"]
    neu = ["It arrived on Tuesday", "The package contains one item", "It is a black box"]
    df = pd.DataFrame({"text": (pos + neg + neu) * 10,
                       "sentiment": (["positive"] * 3 + ["negative"] * 3 + ["neutral"] * 3) * 10})

df["clean"] = df["text"].apply(clean)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"])

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
model = LogisticRegression(max_iter=1000)
model.fit(tfidf.fit_transform(X_train), y_train)

pred = model.predict(tfidf.transform(X_test))
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred))

def predict_sentiment(text):
    return model.predict(tfidf.transform([clean(text)]))[0]

for t in ["I absolutely love this phone!", "This is the worst service ever", "The parcel came today"]:
    print(f"'{t}' -> {predict_sentiment(t).upper()}")
