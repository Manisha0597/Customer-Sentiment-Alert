import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

# Load sentiment model
sentiment = pipeline("sentiment-analysis")

st.set_page_config(page_title="Customer Sentiment Alert", layout="wide")
st.title("📊 Customer Sentiment Alert System")
st.write("Upload reviews/tweets and get instant sentiment with urgency scoring.")

# Text input
user_input = st.text_area("✍️ Enter customer feedback here:", "")

def urgency_score(result):
    if result['label'] == 'NEGATIVE':
        return round(result['score'] * 100, 2)
    return 0

if st.button("Analyze"):
    if user_input.strip() != "":
        result = sentiment(user_input)[0]
        score = urgency_score(result)

        st.subheader("🔎 Analysis Result")
        st.write(f"**Sentiment:** {result['label']}")
        st.write(f"**Confidence:** {result['score']:.2f}")
        st.write(f"**Urgency Score:** {score}")

        # DataFrame for visualization
        df = pd.DataFrame([{
            "Sentiment": result['label'],
            "Confidence": result['score'],
            "Urgency": score
        }])

        st.dataframe(df)

        # Bar chart
        fig, ax = plt.subplots()
        ax.bar(["Urgency Score"], [score])
        ax.set_ylim(0, 100)
        st.pyplot(fig)
    else:
        st.warning("Please enter some feedback to analyze.")
