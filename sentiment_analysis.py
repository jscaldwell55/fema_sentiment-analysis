import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def analyze_sentiment():
    df = pd.read_csv("data/preprocessed_articles.csv")

    with open("models/sentiment_model.pkl", "rb") as file:
        clf = pickle.load(file)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(df["preprocessed_text"])

    tfidf_matrix = vectorizer.transform(df["preprocessed_text"])
    sentiment_pred = clf.predict(tfidf_matrix)

    df["sentiment"] = sentiment_pred
    df.to_csv("data/sentiment_results.csv", index=False)
