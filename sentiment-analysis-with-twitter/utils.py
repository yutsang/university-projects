"""
utils.py
Shared utility functions for the tweet sentiment analysis pipeline.
"""

import os
from typing import Optional

def ensure_dir(path: str) -> None:
    """Create the directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path) 