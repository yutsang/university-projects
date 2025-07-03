"""
feature_engineering.py
Feature engineering utilities for tweet sentiment analysis pipeline.
Requires pandas to be installed in your environment.
"""

import re
import pandas as pd
from typing import Callable

def count_hashtags(text: str) -> int:
    """Count the number of hashtags in a tweet."""
    return len(re.findall(r"#(\w+)", text))

def count_mentions(text: str) -> int:
    """Count the number of mentions in a tweet."""
    return len(re.findall(r"@(\w+)", text))

def count_words(text: str) -> int:
    """Count the number of words in a tweet."""
    return len(text.split())

def add_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Add feature columns for number of words, hashtags, and mentions to the DataFrame.
    Args:
        df: DataFrame with tweet text.
        text_col: Name of the column containing tweet text.
    Returns:
        DataFrame with new feature columns.
    """
    df["NumWords"] = df[text_col].apply(count_words)
    df["NumHashtags"] = df[text_col].apply(count_hashtags)
    df["NumMentions"] = df[text_col].apply(count_mentions)
    return df 