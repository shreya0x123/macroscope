import pandas as pd

df = pd.read_csv("macro_news_with_sentiment.csv")
print("Columns in CSV:", df.columns.tolist())
print(df.head())
