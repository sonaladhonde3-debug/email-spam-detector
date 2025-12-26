import streamlit as st
import pickle

with open('spam_classifier_model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)
st.title("Email Spam Classifier")
st.write("Enter the email content below to check if it's Spam or Not Spam.")
user_input = st.text_area("Email Content")
if st.button('check'):
    if user_input.strip() == "":
        st.warning("Please enter some email content.")
        email_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(email_tfidf)