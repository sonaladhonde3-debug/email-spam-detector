import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.naive_bayes import MultinomialNB

MODEL_PATH = "model/spam_model.pkl"

@st.cache_resource
def train_and_load_model():
    data = pd.read_csv("spam.csv")
    data["v1"] = data["v1"].map({"ham": 0, "spam": 1})

    X = data["v2"]
    y = data["v1"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer


# Load or train model
model, vectorizer = train_and_load_model()

# Streamlit UI
st.title("Email Spam Detection System")
st.write("Enter an email message to check whether it is spam or not.")

email_text = st.text_area("Email content")

if st.button("Check"):
    if email_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        email_tfidf = vectorizer.transform([email_text])
        prediction = model.predict(email_tfidf)

        if prediction[0] == 1:
            st.error("This email is classified as SPAM.")
        else:
            st.success("This email is NOT spam.")
