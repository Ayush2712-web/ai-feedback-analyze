from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')  # more accurate for real reviews

def get_embeddings(reviews):
    return model.encode(reviews, show_progress_bar=True)
