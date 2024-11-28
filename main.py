import os
from scripts.preprocess_data import preprocess_data
from scripts.analyze_sentiment import analyze_sentiment
from scripts.visualize_results import visualize_results

# Data preprocessing step
preprocess_data()

# Sentiment analysis step
analyze_sentiment()

# Visualize the results
visualize_results()

# Upload results to Azure Blob Storage
local_files = [
    ("data/results/vader_vader_sentiment_results.csv", "results/vader_sentiment.csv"),
    ("data/results/textblob_vader_sentiment_results.csv", "results/textblob_sentiment.csv"),
]

# # Iterate over the files and upload each to Blob Storage
# for local_file, blob_name in local_files:
#     upload_to_blob(local_file, blob_name)

print("Sentiment analysis completed and results uploaded to Azure Blob Storage.")