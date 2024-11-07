import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

def preprocess_data():
    df = pd.read_csv("data/raw_articles.csv")
    df["clean_content"] = df["content"].apply(clean_text)
    df["tokens"] = df["clean_content"].apply(nltk.word_tokenize)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    df["preprocessed_tokens"] = df["tokens"].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words])

    df.to_csv("data/preprocessed_articles.csv", index=False)
