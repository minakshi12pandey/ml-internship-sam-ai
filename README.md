# Machine Learning Internship Projects: SAM AI Technologies

Three machine learning projects completed for the SAM AI Technologies internship program.

| # | Project | Techniques | Metrics |
|---|---------|-----------|---------|
| 1 | Spam SMS Detection | TF-IDF, Naive Bayes, Logistic Regression | Accuracy, confusion matrix |
| 2 | Sentiment Analysis | Text cleaning, stopword removal, TF-IDF, Logistic Regression | Accuracy, classification report |
| 4 | Credit Card Fraud Detection | Logistic Regression, Random Forest, class weighting | Precision, recall, F1-score |

## Datasets
Download these (e.g., from Kaggle) and place them in the project folder:
- `spam.csv`: SMS Spam Collection
- `reviews.csv`: any reviews/tweets dataset with `text` and `sentiment` columns
- `creditcard.csv`: Credit Card Fraud Detection

If a file is missing, each script falls back to small demo data so it still runs.

## Setup
```bash
pip install pandas scikit-learn
python task1_spam_sms_detection.py
python task2_sentiment_analysis.py
python task4_credit_card_fraud_detection.py
```

## Approach
- **Task 1:** Messages are vectorized with TF-IDF and classified as spam or legitimate with two simple models that are then compared.
- **Task 2:** Text is lowercased, stripped of URLs, mentions and punctuation, and filtered for stopwords. TF-IDF (unigrams and bigrams) feeds a Logistic Regression model that predicts positive, negative or neutral.
- **Task 4:** Fraud is rare, so accuracy is misleading. Class weights handle the imbalance, and models are judged on precision, recall and F1 for the fraud class.

## Results
Fill in the numbers after running each script on the real dataset.

| Project | Model | Result |
|---------|-------|--------|
| Spam SMS Detection | Naive Bayes | Accuracy: 96.9% |
| Spam SMS Detection | Logistic Regression | Accuracy: 96.8% |
| Sentiment Analysis | Logistic Regression | Accuracy: 97% |
| Fraud Detection | Logistic Regression | Precision: [X] / Recall: [X] / F1: [X] |
| Fraud Detection | Random Forest | Precision: [X] / Recall: [X] / F1: [X] |

## Author
Minakshi Pandey | https://www.linkedin.com/in/contactminakshipandey | Internship: SAM AI Technologies
