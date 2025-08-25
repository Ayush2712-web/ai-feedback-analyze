import pandas as pd
from preprocess import clean_text
from embeddings import get_embeddings
from cluster import cluster_embeddings
from sentiment import get_sentiment

# Load reviews
df = pd.read_csv('C:/Users/ASUS/Documents/ai-feedback-analyzer/data/real_reviews.csv')
df['cleaned'] = df['review_text'].apply(clean_text)

# Generate embeddings
embeddings = get_embeddings(df['cleaned'].tolist())

# Cluster reviews
df['cluster'] = cluster_embeddings(embeddings, n_clusters=3)

# Analyze sentiment
df[['sentiment', 'sentiment_score']] = df['cleaned'].apply(lambda x: pd.Series(get_sentiment(x)))

# Show results
print(df[['review_text', 'cluster', 'sentiment', 'sentiment_score']])
