import pandas as pd
import re

def preprocess_data():
    # Load raw data
    df = pd.read_csv("data/raw/twitter_sentiment.csv")
    
    # Clean the text
    # Check if the value is a string before applying regex
    df['cleaned_text'] = df['tweet'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', str(x)))  # Remove URLs
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'@\S+', '', str(x)))  # Remove mentions
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))  # Remove special characters
    
    # Save cleaned data
    df.to_csv("data/cleaned/cleaned_twitter_sentiment.csv", index=False)
    print("Cleaned data saved to data/cleaned/cleaned_twitter_sentiment.csv")
