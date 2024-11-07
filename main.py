import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

# Import only collect_data_gdelt for GDELT data collection
from data_collection import collect_data_gdelt
from data_preprocessing import preprocess_data
from feature_extraction import extract_features
from model_training import train_model
from sentiment_analysis import analyze_sentiment

def generate_visualizations(df):
    # Sentiment Distribution
    sentiment_counts = df["sentiment"].value_counts()
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind="bar")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Distribution")
    plt.savefig("visualizations/sentiment_distribution.png")
    plt.close()

    # Word Cloud
    text = " ".join(df["clean_content"])
    wordcloud = WordCloud(width=800, height=600, background_color="white").generate(text)
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("visualizations/word_cloud.png")
    plt.close()

def main():
    # Collect data from GDELT API with the specified query and max number of records
    collect_data_gdelt(query="climate change", max_records=10)  # GDELT call
    
    # Run the rest of the data pipeline
    preprocess_data()
    extract_features()
    train_model()
    analyze_sentiment()

    # Load the preprocessed articles and results for generating visualizations
df = pd.read_csv("data/preprocessed_articles.csv", error_bad_lines=False, warn_bad_lines=True)
df = pd.read_csv("data/preprocessed_articles.csv", delimiter=',')

    # Generate visualizations based on the processed data
generate_visualizations(df)

if __name__ == "__main__":
    main()
