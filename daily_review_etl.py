import pandas as pd
import sqlite3
from textblob import TextBlob
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os
import time
from tqdm import tqdm

# Output folder path
output_dir = r"C:\Users\s698441\OneDrive - Sanlam Life Insurance Limited\Desktop\FNB\Review_Output"
os.makedirs(output_dir, exist_ok=True)

# Input file paths
file_noluthando_branch = r"C:\Users\s698441\OneDrive - Sanlam Life Insurance Limited\Desktop\FNB\Branch List Noluthando(1).csv"
file_sizwe_branch = r"C:\Users\s698441\OneDrive - Sanlam Life Insurance Limited\Desktop\FNB\branch list Sizwe(1).csv"
file_noluthando_reviews = r"C:\Users\s698441\OneDrive - Sanlam Life Insurance Limited\Desktop\FNB\data capture Noluthando(1).csv"
file_sizwe_reviews = r"C:\Users\s698441\OneDrive - Sanlam Life Insurance Limited\Desktop\FNB\data capture Sizwe(1).csv"

# Step 1: Load CSVs
print("Reading CSV files...")
df_noluthando = pd.read_csv(file_noluthando_branch)
df_sizwe = pd.read_csv(file_sizwe_branch)
df_reviews_noluthando = pd.read_csv(file_noluthando_reviews, encoding="ISO-8859-1")
df_reviews_sizwe = pd.read_csv(file_sizwe_reviews, encoding="ISO-8859-1")

# Step 2: Combine branch lists
print("Combining branch lists...")
df_branches = pd.concat([df_noluthando, df_sizwe], ignore_index=True)
df_branches = df_branches.drop_duplicates(subset=["Branch Location"])
df_branches.columns = [col.strip() for col in df_branches.columns]

# Step 3: Combine review data
print("Combining review data...")
df_reviews = pd.concat([df_reviews_noluthando, df_reviews_sizwe], ignore_index=True)
df_reviews.columns = [col.strip() for col in df_reviews.columns]
df_reviews = df_reviews.drop_duplicates(subset=["Branch Location", "Reviewer", "Comment"])
df_reviews["Rating"] = pd.to_numeric(df_reviews["Rating"], errors="coerce")
df_reviews = df_reviews.dropna(subset=["Branch Location", "Rating"])

# Step 4: Merge reviews with location info
print("Merging location info into reviews...")
df_final = pd.merge(df_reviews, df_branches, on="Branch Location", how="left")

# Step 4.1: Split geo-coordinates
print("Splitting Geo-Coordinates...")
if "Geo-Coordinates" in df_final.columns:
    df_final[["Latitude", "Longitude"]] = df_final["Geo-Coordinates"].str.split(",", expand=True)
    df_final["Latitude"] = pd.to_numeric(df_final["Latitude"], errors="coerce")
    df_final["Longitude"] = pd.to_numeric(df_final["Longitude"], errors="coerce")

# Step 4.1.1: Reverse Geocode to Get Province
print("Reverse geocoding with province extraction...")
geolocator = Nominatim(user_agent="fnb_review_locator", timeout=10)
geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1, max_retries=5)

geocode_cache = {}
failed_coords = []
province_list = []

for idx, row in tqdm(df_final.iterrows(), total=df_final.shape[0]):
    lat, lon = row["Latitude"], row["Longitude"]
    if pd.notnull(lat) and pd.notnull(lon):
        key = f"{lat},{lon}"
        if key in geocode_cache:
            province = geocode_cache[key]
        else:
            try:
                location = geocode((lat, lon), language='en')
                province = location.raw['address'].get('state') or location.raw['address'].get('region')
                geocode_cache[key] = province
            except Exception as e:
                print(f"Error for ({lat}, {lon}): {e}")
                province = None
                failed_coords.append({"index": idx, "Latitude": lat, "Longitude": lon, "reason": str(e)})
    else:
        province = None
    province_list.append(province)

df_final["province"] = province_list

# Step 4.2: Sentiment Analysis with Keyword Matching
print("Performing sentiment analysis with keyword extraction...")

positive_keywords = ["excellent", "good", "great", "amazing", "wonderful", "friendly", "helpful", "quick", "efficient"]
negative_keywords = ["bad", "terrible", "awful", "slow", "rude", "unhelpful", "poor", "horrible"]
neutral_keywords = ["okay", "average", "mediocre", "not bad", "decent", "fine"]

def classify_sentiment_with_keywords(text):
    if pd.isna(text):
        return "Neutral", ""
    text_lower = text.lower()
    matched_keywords = []

    matched_keywords += [word for word in positive_keywords if word in text_lower]
    matched_keywords += [word for word in negative_keywords if word in text_lower]
    matched_keywords += [word for word in neutral_keywords if word in text_lower]

    polarity = TextBlob(text).sentiment.polarity

    if matched_keywords:
        sentiment = "Neutral"
        if any(word in positive_keywords for word in matched_keywords):
            sentiment = "Positive"
        if any(word in negative_keywords for word in matched_keywords):
            sentiment = "Negative"
    else:
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

    return sentiment, ", ".join(matched_keywords)

df_final[["Sentiment", "SentimentKeywords"]] = df_final["Comment"].apply(
    lambda x: pd.Series(classify_sentiment_with_keywords(x))
)

# Step 4.3: Save to CSVs
print("Saving results to CSV files...")

df_final.to_csv(os.path.join(output_dir, 'ReviewsWithSentiment.csv'), index=False)
df_branches.to_csv(os.path.join(output_dir, 'branches.csv'), index=False)
df_reviews.to_csv(os.path.join(output_dir, 'reviews_raw.csv'), index=False)
df_final.to_csv(os.path.join(output_dir, 'reviews_cleaned.csv'), index=False)

# Step 5: Save all data to SQLite DB
print("Saving to SQLite database...")
db_path = os.path.join(output_dir, "branch_reviews.db")
conn = sqlite3.connect(db_path)

df_branches.to_sql("branches", conn, if_exists="replace", index=False)
df_reviews.to_sql("reviews_raw", conn, if_exists="replace", index=False)
df_final.to_sql("reviews_cleaned", conn, if_exists="replace", index=False)
df_final.to_sql("reviews_with_sentiment", conn, if_exists="replace", index=False)

conn.close()
#print(f"All data saved to database at: {db_path}")
#print("✅ Script complete.")
