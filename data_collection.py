import requests
import pandas as pd

def collect_data_gdelt(query="climate change", max_records=10):
    """
    Collects news data from the GDELT API based on the provided query and number of records.
    Saves the results as a CSV file.
    
    Args:
        query (str): The search query to find articles (e.g., "climate change").
        max_records (int): The maximum number of articles to retrieve (default is 10).
    """
    # GDELT API URL
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={query}&mode=artlist&maxrecords={max_records}&format=json"
    
    # Send GET request to GDELT API
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve data from GDELT API")
        return
    
    # Convert response JSON to Python dictionary
    articles = response.json()
    
    # Check if 'articles' field exists
    if "articles" not in articles:
        print("No articles found")
        return
    
    # Prepare data for DataFrame
    data = []
    for article in articles["articles"]:
        data.append({
            "title": article.get("title", "No Title"),
            "url": article.get("url", "No URL"),
            "date": article.get("seendate", "No Date"),
            "source": article.get("domain", "No Source"),
            "language": article.get("language", "No Language")
        })
    
    # Create a DataFrame from the list of articles
    df = pd.DataFrame(data)
    
    # Save DataFrame to CSV
    df.to_csv("data/preprocessed_articles.csv", index=False)
    print(f"Collected {len(df)} articles and saved to 'data/preprocessed_articles.csv'")

# Example usage:
collect_data_gdelt(query="climate change", max_records=10)
