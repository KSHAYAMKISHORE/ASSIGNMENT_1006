import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

def plot_bar_chart(df, column, title, output_filename):
    # Simplify sentiment into categories
    def categorize_sentiment(score):
        if score > 0:
            return 'Positive'
        elif score < 0:
            return 'Negative'
        else:
            return 'Neutral'

    # Create a new column for categorized sentiment
    df['sentiment_category'] = df[column].apply(categorize_sentiment)

    # Get the sentiment counts
    sentiment_counts = df['sentiment_category'].value_counts()

    # Plot the bar chart
    sentiment_counts.plot(kind='bar', color=['green', 'blue', 'red'], alpha=0.7)
    plt.title(title)
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)  # Keep x-axis labels horizontal
    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig(f"outputs/{output_filename}")
    plt.show()

def plot_wordcloud(df, text_column, output_filename):
    # Join all text into one large string
    text = " ".join(df[text_column].dropna())  # Drop NaN values just in case
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(f"outputs/{output_filename}")
    plt.show()

def visualize_results():
    # Load sentiment analysis results
    df = pd.read_csv("data/results/textblob_sentiment_results.csv")

    # Check for the existence of 'textblob_sentiment'
    if 'textblob_sentiment' not in df.columns:
        print("TextBlob sentiment column not found. Adding sentiment analysis...")

        # Define a function to calculate TextBlob sentiment
        def textblob_sentiment(text):
            if isinstance(text, str):
                return TextBlob(text).sentiment.polarity
            else:
                return 0.0

        # Apply TextBlob sentiment analysis to the 'cleaned_text' column
        df['textblob_sentiment'] = df['cleaned_text'].apply(textblob_sentiment)

        # Save the updated dataframe with the new sentiment column
        df.to_csv("data/results/textblob_sentiment_results.csv", index=False)

    # Plot bar chart for simplified TextBlob sentiment distribution
    plot_bar_chart(df, 'textblob_sentiment', "Simplified TextBlob Sentiment Distribution", "textblob_sentiment_bar_chart.png")

    # Plot Wordcloud
    plot_wordcloud(df, 'cleaned_text', "wordcloud.png")

if __name__ == "__main__":
    visualize_results()
