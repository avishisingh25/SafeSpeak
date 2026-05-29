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

st.write(f"Model Label: {label}")
st.write(f"Confidence: {score*100:.2f}%")

if label.lower() == "toxic" and score > 0.80:

    st.error("⚠️ Potential Hate Speech / Abuse Detected")

    st.progress(int(score * 100))

    st.write("Risk Level: High")

else:

    st.success("✅ Safe Content")

    st.progress(int(score * 100))

    st.write("Risk Level: Low")