# Email Spam Detection System

This project is a beginner-level Email Spam Detection System built using
Machine Learning and Natural Language Processing techniques. The system
classifies email text as either spam or not spam.

---

## Overview

Email spam detection is a binary text classification problem. This project
uses TF-IDF for text feature extraction and a Naive Bayes classifier to
identify spam emails. A Streamlit web interface is provided for real-time
prediction.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Natural Language Processing (TF-IDF)

---

## Project Structure
│
├── data/
│ └── spam.csv
├── model/
│ └── spam_model.pkl
├── train_model.py
├── app.py
├── requirements.txt
└── README.md

---

## How It Works

1. Email text is preprocessed and converted into numerical features using
   TF-IDF.
2. A Naive Bayes classifier is trained on labeled email data.
3. The trained model predicts whether new email messages are spam or not.
4. A Streamlit web interface allows users to test the model interactively.

---

## Installation and Execution

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt

Step 2: Train the Model
python train_model.py

Step 3: Run the Application
streamlit run app.py

Example

Input:

Congratulations! You have won a free prize.


Output:

Spam

-> Learning Outcomes

Understanding of supervised machine learning

Text preprocessing and feature extraction using TF-IDF

Implementation of Naive Bayes for text classification

Deployment of ML models using Streamlit