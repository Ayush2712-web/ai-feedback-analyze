import contractions
from textblob import TextBlob
import emoji
import re

def clean_text(text):
    text = contractions.fix(text)           # Expand contractions
    text = emoji.demojize(text)             # Convert emojis to words
    text = text.lower()
    text = re.sub(r"http\S+", "", text)    # Remove URLs
    text = re.sub(r"[^0-9a-zA-Z\s]", "", text)  # Remove special characters
    text = str(TextBlob(text).correct())   # Optional spell correction
    return text
