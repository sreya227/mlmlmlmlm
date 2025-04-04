import pandas as pd
import numpy as np
import nltk
import string
import joblib
from nltk.corpus import stopwords, sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Ensure necessary nltk data is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')

# Load dataset
df = pd.read_csv("train.csv")

# Text preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word not in string.punctuation]
    return " ".join(words)

df['processed_text'] = df['Statement'].astype(str).apply(preprocess_text)

# Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['processed_text'])

# Sentiment Analysis Feature
def get_sentiment_score(text):
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    sentiment_score = 0
    for word, tag in pos_tags:
        if tag.startswith('N'):
            synsets = list(swn.senti_synsets(word, 'n'))
        elif tag.startswith('V'):
            synsets = list(swn.senti_synsets(word, 'v'))
        elif tag.startswith('R'):
            synsets = list(swn.senti_synsets(word, 'r'))
        elif tag.startswith('J'):
            synsets = list(swn.senti_synsets(word, 'a'))
        else:
            continue
        if synsets:
            sentiment_score += synsets[0].pos_score() - synsets[0].neg_score()
    return sentiment_score

df['sentiment'] = df['processed_text'].apply(get_sentiment_score)

# Combine TF-IDF with Sentiment Score
X_features = np.hstack((X_tfidf.toarray(), df[['sentiment']].values))
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Model Pipeline
model = Pipeline([
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save Model & Vectorizer
joblib.dump(model, 'final_model.sav')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("✔ Model & vectorizer saved successfully!")
