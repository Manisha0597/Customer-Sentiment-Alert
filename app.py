import streamlit as st
from transformers import pipeline
import math

# Load model
sentiment_pipeline = pipeline('sentiment-analysis')

def urgency_score(sentiment, reach=1, is_mention=False):
    label = sentiment.get('label', 'NEUTRAL').upper()
    score = sentiment.get('score', 0.0)
    base = 0.0
    if label in ('NEGATIVE','LABEL_0','NEG'):
        base = 0.6 + 0.4*score
    elif label in ('POSITIVE','LABEL_1','POS'):
        base = 0.1*score
    else:
        base = 0.2
    reach_factor = math.log1p(reach)/10.0
    mention_factor = 0.15 if is_mention else 0.0
    urgency = min(1.0, base + reach_factor + mention_factor)
    return round(urgency,3)

# ---- UI ----
st.title("🚨 Customer Sentiment Alert")
feedback = st.text_area("Enter customer feedback:")
reach = st.number_input("Reach (No. of people affected)", min_value=1, value=10)
is_mention = st.checkbox("Is this a direct mention?")

if st.button("Analyze"):
    sentiment = sentiment_pipeline(feedback)[0]
    urgency = urgency_score(sentiment, reach, is_mention)
    st.write("### Results")
    st.write(f"**Sentiment:** {sentiment['label']} (Score: {sentiment['score']:.3f})")
    st.write(f"**Urgency Score:** {urgency}")
