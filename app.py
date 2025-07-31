import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv("macro_news_with_sentiment.csv")
market_df = pd.read_csv("market_returns.csv")

# Convert date columns to datetime
df["date"] = pd.to_datetime(df["date"])
market_df["Date"] = pd.to_datetime(market_df["Date"])

# Merge sentiment and market return data
merged = pd.merge(df, market_df, left_on="date", right_on="Date", how="inner")

# Sidebar filters
st.sidebar.title("Filters")
selected_country = st.sidebar.multiselect("Country", df["country"].unique(), default=df["country"].unique())
selected_query = st.sidebar.multiselect("Query", df["query"].unique(), default=df["query"].unique())

# Apply filters
filtered_df = merged[(merged["country"].isin(selected_country)) & (merged["query"].isin(selected_query))]

# App title
st.title("📈 Macroscope: Market Sentiment + ML Forecast")
st.markdown("_AI-powered dashboard for macroeconomic sentiment and market predictions._")

# Sentiment Distribution
st.subheader("Sentiment Distribution")
sentiment_counts = filtered_df["sentiment_label"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]
fig = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment", title="Sentiment Label Distribution")
st.plotly_chart(fig)

# Sentiment Score Over Time
st.subheader("Average Sentiment Over Time")
score_plot = filtered_df.groupby("date")["sentiment_mean"].mean().reset_index()
fig2 = px.line(score_plot, x="date", y="sentiment_mean", title="Sentiment Score Over Time")
st.plotly_chart(fig2)

# ML Target Plot
st.subheader("ML Target: Did Market Go Up?")
st.write("`Target = 1` means market went up, `0` means down.")
fig3 = px.scatter(filtered_df, x="date", y="daily_return", color="target",
                  labels={"target": "Market Up (1) or Down (0)", "daily_return": "Return"},
                  title="Daily Return vs Sentiment-based Prediction")
st.plotly_chart(fig3)

# Data Preview
st.subheader("📄 Data Preview")
st.dataframe(filtered_df[["date", "query", "sentiment_mean", "sentiment_label", "daily_return", "target"]].sort_values("date", ascending=False))
