import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="SafeSpeak",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SafeSpeak")

st.write(
    "AI Powered Hate Speech & Gender-Based Abuse Detector"
)

text = st.text_area(
    "Enter a social media comment"
)

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter a comment.")

    else:

        with st.spinner("Analyzing..."):

            classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert"
            )

            result = classifier(text)

        label = result[0]["label"]
        score = result[0]["score"]

        st.subheader("Analysis Result")

        st.write(f"Classification: {label}")

        st.write(
            f"Confidence: {score*100:.2f}%"
        )