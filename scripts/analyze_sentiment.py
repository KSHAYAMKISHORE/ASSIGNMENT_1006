import pandas as pd
from textblob import TextBlob

def textblob_sentiment(text):
    if isinstance(text, str):
        # Get polarity using TextBlob
        blob = TextBlob(text)
        return blob.sentiment.polarity
    else:
        return 0.0  # Return neutral sentiment for non-string inputs

def analyze_sentiment():
    # Load cleaned data
    df = pd.read_csv("data/cleaned/cleaned_twitter_sentiment.csv")
    
    # Apply TextBlob sentiment analysis
    df['textblob_sentiment'] = df['cleaned_text'].apply(textblob_sentiment)
    
    # Save the results with TextBlob sentiment
    df.to_csv("data/results/textblob_sentiment_results.csv", index=False)
    print("Sentiment analysis (TextBlob) saved to data/results/textblob_sentiment_results.csv")

    # Optionally, you can also return the DataFrame for further analysis
    return df

if __name__ == "__main__":
    analyze_sentiment()
