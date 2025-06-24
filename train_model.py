import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample training data (replace with your actual dataset)
data = pd.DataFrame({
    'text': ['win money now', 'free prize click', 'hello friend', 'meeting tomorrow'],
    'label': [1, 1, 0, 0]  # 1=spam, 0=ham
})

# Feature extraction
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['text'])
y = data['label']

# Train model
model = MultinomialNB()
model.fit(X, y)  # THIS IS THE CRITICAL STEP

# Save models
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model1.pkl', 'wb'))

print("Model trained and saved successfully!")