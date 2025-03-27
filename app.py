import streamlit as st
import joblib
import re

# Load the trained model and vectorizer
model = joblib.load("twitter_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text

# Function to predict sentiment
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    sentiment_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    return sentiment_map[prediction]

# Streamlit UI
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet below to analyze its sentiment:")
st.subheader("Example tweets to try:")

st.markdown("> 1. overpromise and underdelivery – that pithy summary the economic outcome the last five years under approach the general election")
st.markdown("> 2. will these channels say modi also scared contests from two seats even propoganda needs little decency")
st.markdown("> 3. farmers' welfare about 474 farmers get second installment from next month the centre announced the 75000crore scheme")
st.markdown("> 4. I hate using this new AI tool! It's awfull.")
st.markdown("> 5. I love using this new AI tool! It's amazing.")

user_input = st.text_area("Tweet Text", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.write(f"### Sentiment: {sentiment}")
    else:
        st.write("Please enter a tweet to analyze.")
