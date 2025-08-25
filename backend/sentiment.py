from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis")  # can use fine-tuned model if available

def get_sentiment(text):
    result = sentiment_analyzer(text)[0]
    label = result['label']
    score = result['score']
    if score < 0.7:
        label = "NEUTRAL"
    return label, score
