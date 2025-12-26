import pandas as pd
import pickle
#It saves trained models to disk so you don’t retrain every time.

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#Converts text → numbers
from sklearn.naive_bayes import MultinomialNB
#Perfect for text classification

data = pd.read_csv('spam.csv',delimiter=',', encoding='latin-1')

data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})
X = data['v2']
y = data['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english')
#stop_words="english" removes useless words:“the”, “is”, “and”, “to”
#These add noise, not meaning
X_train_tfidf = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
with open('spam_classifier_model.pkl', 'wb') as f:
    pickle.dump((model,vectorizer), f)
#Saves the model and vectorizer to a file named spam_classifier_model.pkl
#wb means write binary
print("Model trained and saved as spam_classifier_model.pkl")
