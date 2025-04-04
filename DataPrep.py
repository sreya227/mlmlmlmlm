import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatization & stopword removal
    return text

# Load dataset
df = pd.read_csv('train.csv')

# Apply cleaning function
df['cleaned_text'] = df['Statement'].astype(str).apply(clean_text)

# Save cleaned dataset
df.to_csv('train_cleaned.csv', index=False, encoding='utf-8')
print("✔ Data preprocessing completed! Saved as 'train_cleaned.csv'")
