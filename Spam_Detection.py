import streamlit as st
from joblib import load
import numpy as np

# Load the saved models and vectorizer
st.title("Spam Message Classifier")
st.write("This application uses multiple machine learning models to classify messages as **Spam** or **Ham**.")

# Load the saved models
models = load("models_Spam.joblib")
voting_clf = models["Voting Classifier"]
vectorizer = models["Vectorizer"]

# Input form
st.header("Enter a Message for Prediction")
user_message = st.text_area("Type your message below:")

if st.button("Classify"):
    if user_message.strip():
        # Preprocess the message using the vectorizer
        message_vectorized = vectorizer.transform([user_message])
        
        # Predict using the Voting Classifier
        prediction = voting_clf.predict(message_vectorized)
        result = "Spam" if prediction[0] == 1 else "Ham"
        
        # Display the result
        st.subheader("Prediction Result")
        st.write(f"The message is classified as: **{result}**")
        
        # Optionally show probabilities if available (soft voting)
        if hasattr(voting_clf, "predict_proba"):
            probabilities = voting_clf.predict_proba(message_vectorized)
            st.write("Prediction Probabilities:")
            st.write(f"Ham: {probabilities[0][0]*100:.2f}%")
            st.write(f"Spam: {probabilities[0][1]*100:.2f}%")
    else:
        st.warning("Please enter a message before clicking 'Classify'.")
