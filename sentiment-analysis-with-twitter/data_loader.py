"""
data_loader.py
Module for loading and cleaning tweet data for sentiment analysis pipeline.
Requires pandas to be installed in your environment.
"""

import re
from typing import List
import pandas as pd

CLEAN_COLUMNS = [
    "timestamp_epochs", "tweet_url", "links", "has_media", "img_urls", "video_url", "parent_tweet_id", "reply_to_users"
]

def clean_text(text: str) -> str:
    """Clean tweet text by removing URLs, mentions, special characters, and extra whitespace."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = " ".join(text.split())
    return text


def load_and_clean_data(csv_path: str, drop_columns: List[str] = CLEAN_COLUMNS) -> pd.DataFrame:
    """
    Load tweet data from a CSV, remove duplicates, irrelevant columns, retweets, and clean text.
    Args:
        csv_path: Path to the CSV file.
        drop_columns: List of columns to drop if present.
    Returns:
        Cleaned pandas DataFrame.
    """
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    df.drop_duplicates(subset="text", inplace=True)
    for col in drop_columns:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    df["text"].fillna("", inplace=True)
    df = df[~df["text"].str.startswith("RT")]
    df["text"] = df["text"].astype(str).apply(clean_text)
    return df 