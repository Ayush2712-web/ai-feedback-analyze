import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from backend.preprocess import clean_text
from backend.embeddings import get_embeddings
from backend.cluster import cluster_embeddings
from backend.sentiment import get_sentiment

st.title("AI Customer Feedback Analyzer")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload CSV with 'review_text' column", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Preprocess
    df['cleaned'] = df['review_text'].apply(clean_text)
    
    # Embeddings + clustering
    embeddings = get_embeddings(df['cleaned'].tolist())
    df['cluster'] = cluster_embeddings(embeddings, n_clusters=3)
    
    # Sentiment
    df[['sentiment', 'sentiment_score']] = df['cleaned'].apply(lambda x: pd.Series(get_sentiment(x)))
    
    st.success("Analysis Complete!")
    
    # Step 2: Show overall stats
    st.subheader("Cluster Distribution")
    st.bar_chart(df['cluster'].value_counts())
    
    st.subheader("Sentiment Distribution")
    st.bar_chart(df['sentiment'].value_counts())
    
    # Step 3: Show reviews per cluster
    st.subheader("Reviews by Cluster")
    cluster_options = df['cluster'].unique()
    selected_cluster = st.selectbox("Select Cluster", cluster_options)
    
    st.write(df[df['cluster'] == selected_cluster][['review_text', 'sentiment', 'sentiment_score']])
    
    # Step 4: Optional: Suggest fixes
    st.subheader("Suggested Fixes (basic)")
    cluster_reviews = df[df['cluster'] == selected_cluster]['cleaned'].tolist()
    if st.button("Generate Suggested Fixes"):
        # Simple keyword-based suggestions
        suggestions = []
        for review in cluster_reviews:
            if "login" in review or "error" in review:
                suggestions.append("Check login functionality and error handling.")
            elif "delivery" in review:
                suggestions.append("Review delivery process for errors or delays.")
            elif "customer service" in review:
                suggestions.append("Train staff for better customer interaction.")
        suggestions = list(set(suggestions))
        for s in suggestions:
            st.write(f"- {s}")
