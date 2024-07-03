import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Load the model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load('vectorizer.joblib')

def predict_spam(email_text):
    # Vectorize the input text
    email_vector = vectorizer.transform([email_text])
    # Predict using the loaded model
    prediction = model.predict(email_vector)
    return prediction[0]

# Streamlit UI
def main():
    st.title("Email Spam Detector")
    email_text = st.text_area("Enter your email text here:")
    if st.button("Predict"):
        prediction = predict_spam(email_text)
        if prediction == 1:
            st.error("This email is classified as SPAM.")
        else:
            st.success("This email is classified as NOT SPAM.")

if __name__ == "__main__":
    main()