import requests
import pandas as pd
import time
from datetime import datetime, timedelta

API_KEY = "pub_ba89de877a1046eb985e3d1bc1438d2b"  
QUERIES = ["inflation", "interest rates", "recession", "gdp"]
COUNTRIES = ["us", "in", "gb"]
START_DATE = datetime.today() - timedelta(days=30)
END_DATE = datetime.today()

all_articles = []

print("🔎 Fetching news from NewsData.io")

for query in QUERIES:
    for delta in range((END_DATE - START_DATE).days + 1):
        date = START_DATE + timedelta(days=delta)
        date_str = date.strftime("%Y-%m-%d")

        for country in COUNTRIES:
            url = "https://newsdata.io/api/1/news"
            params = {
                "apikey": API_KEY,
                "q": query,
                "country": country,
                "language": "en",
                "from_date": date_str,
                "to_date": date_str
            }

            try:
                response = requests.get(url, params=params, timeout=10)
                data = response.json()  # ✅ FIX: ensure response is parsed correctly

                articles = data.get("results", [])
                if not articles:
                    print(f"⚠️ No articles for {query} on {date_str} ({country})")
                    continue

                for article in articles:
                    all_articles.append({
                        "date": date_str,
                        "query": query,
                        "country": country,
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "content": article.get("content", ""),
                        "source_url": article.get("link", "")
                    })

                print(f"✅ {len(articles)} articles fetched for {query} on {date_str} ({country})")
                time.sleep(1.5)  # To respect rate limits

            except requests.exceptions.RequestException as e:
                print(f"❌ Network error for {query} on {date_str} ({country}): {e}")
            except Exception as e:
                print(f"❌ Error fetching {query} on {date_str} ({country}): {e}")

# Save to CSV
if all_articles:
    df = pd.DataFrame(all_articles)
    df.to_csv("raw_macro_news.csv", index=False)
    print(f"\n📁 Saved {len(df)} articles to raw_macro_news.csv")
else:
    print("❌ No data collected. Try broader queries or check API key.")
