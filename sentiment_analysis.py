import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load news data
df = pd.read_csv("macro_news.csv")

# Convert publishedAt to date
df["date"] = pd.to_datetime(df["publishedAt"]).dt.date

# Run sentiment analysis
analyzer = SentimentIntensityAnalyzer()
df["text"] = df["title"].fillna('') + " " + df["description"].fillna('')
df["sentiment_score"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

def label_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"

df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

# Save results
df.to_csv("macro_news_with_sentiment.csv", index=False)
print("✅ Sentiment analysis complete and saved to macro_news_with_sentiment.csv")
