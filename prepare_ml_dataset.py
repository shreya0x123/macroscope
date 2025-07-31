import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load data
sentiment_df = pd.read_csv("macro_news_with_sentiment.csv")
returns_df = pd.read_csv("market_returns.csv")

# Parse dates
sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
returns_df["Date"] = pd.to_datetime(returns_df["Date"])
returns_df.rename(columns={"Date": "date"}, inplace=True)

# Aggregate sentiment scores by date
agg_sentiment = sentiment_df[["date", "sentiment_mean"]].drop_duplicates().reset_index()

# Merge with market returns
merged_df = pd.merge(agg_sentiment, returns_df, on="date", how="inner")

# Define target: 1 if return > 0, else 0
merged_df["target"] = (merged_df["daily_return"] > 0).astype(int)

# ==============================
# 🚀 Feature Engineering Begins
# ==============================

# Sort by date to ensure lag logic
merged_df = merged_df.sort_values("date")

# Lag features
merged_df["sentiment_mean_lag_1"] = merged_df["sentiment_mean"].shift(1)
merged_df["sentiment_mean_lag_2"] = merged_df["sentiment_mean"].shift(2)

# Rolling window stats
merged_df["sentiment_3day_mean"] = merged_df["sentiment_mean"].rolling(window=3).mean()
merged_df["sentiment_3day_std"] = merged_df["sentiment_mean"].rolling(window=3).std()

# Interaction feature
merged_df["sentiment_volatility_score"] = (
    merged_df["sentiment_3day_mean"] * merged_df["sentiment_3day_std"]
)

# Drop rows with NaNs due to lags and rolling stats
merged_df.dropna(inplace=True)

# Prepare features and labels
features = [
    "sentiment_mean",
    "sentiment_mean_lag_1",
    "sentiment_mean_lag_2",
    "sentiment_3day_mean",
    "sentiment_3day_std",
    "sentiment_volatility_score"
]
X = merged_df[features]
y = merged_df["target"]

# Diagnostics
print("\n🔎 Dataset diagnostics before train-test split:")
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Number of NaNs in features: {X.isna().sum().sum()}")
print(f"Number of NaNs in labels: {y.isna().sum()}")
print(merged_df[["date", "sentiment_mean", "target"]].head())

if len(X) < 10:
    print("❌ ERROR: Not enough data for reliable training. Please add more data.")
    exit()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\n✅ Model Accuracy: {round(accuracy * 100, 2)}%")

# Show feature importances
importances = pd.Series(model.feature_importances_, index=features)
importances = importances.sort_values(ascending=False)
print("\n📊 Feature Importances:")
print(importances)
