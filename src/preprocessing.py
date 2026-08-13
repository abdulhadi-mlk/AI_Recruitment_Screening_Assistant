import re
from typing import Optional


def preprocess_text(text: Optional[str]) -> str:
    """Clean and normalize raw text for matching and analysis."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
