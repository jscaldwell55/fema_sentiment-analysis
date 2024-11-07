import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features():
    df = pd.read_csv("data/preprocessed_articles.csv")
    df["preprocessed_text"] = df["preprocessed_tokens"].apply(lambda x: " ".join(x))

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["preprocessed_text"])

    pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names()).to_csv("data/tfidf_features.csv", index=False)
