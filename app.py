# app.py
import streamlit as st
import joblib
import numpy as np

# Load the saved model and vectorizer
model = joblib.load("model.pkl")  # Loads the pre-trained model
vectorizer = joblib.load("vectorizer.pkl")  # Loads the TF-IDF vectorizer or CountVectorizer

# Define the prediction function
def predict_news(text):
    # Transform the input text using the vectorizer
    transformed_text = vectorizer.transform([text])
    
    # Predict using the trained model
    prediction = model.predict(transformed_text)
    
    # Return "Fake" or "Real" based on the prediction
    return "Fake" if prediction[0] == 0 else "Real"

# Streamlit UI Configuration
st.set_page_config(page_title="Fake News Detection", layout="centered")

# Add title to the app
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article or statement below to check if it's **real** or **fake**.")

# Input field for news text
user_input = st.text_area("✍️ Paste the news content here:", height=200)

# When the user presses the 'Predict' button
if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some news text to analyze.")
    else:
        # Get the prediction
        result = predict_news(user_input)
        # Display the result
        st.success(f"The news is **{result.upper()}**.")
