from transformers import pipeline

# Load HuggingFace sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_feedback(text):
    result = sentiment_pipeline(text)
    return result

if __name__ == "__main__":
    # Example feedbacks
    feedbacks = [
        "I love the product, it's amazing!",
        "This is terrible, I hate the service.",
        "The app is okay, but it could be faster."
    ]

    for fb in feedbacks:
        print(f"Feedback: {fb}")
        print(f"Sentiment: {analyze_feedback(fb)}\n")
