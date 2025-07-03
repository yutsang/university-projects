"""
sentiment_analysis.py
Sentiment analysis utilities for tweet sentiment analysis pipeline.
Requires pandas and nltk to be installed in your environment.
"""

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Optional
import nltk

def add_sentiment_scores(df: pd.DataFrame, text_col: str = "text", sentiment_col: str = "Sentiment") -> pd.DataFrame:
    """
    Add sentiment scores to a DataFrame using NLTK VADER.
    Args:
        df: DataFrame with tweet text.
        text_col: Name of the column containing tweet text.
        sentiment_col: Name of the column to store sentiment scores.
    Returns:
        DataFrame with sentiment scores.
    """
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    df[sentiment_col] = df[text_col].apply(lambda x: sia.polarity_scores(x)["compound"])
    return df 