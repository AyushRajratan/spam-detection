import streamlit as st
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize NLTK
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path = [nltk_data_path]

try:
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)
except Exception as e:
    st.error(f"NLTK initialization failed: {str(e)}")
    st.stop()


# Load models with verification
@st.cache_resource
def load_models():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model1.pkl', 'rb'))

        # Critical model verification
        if not hasattr(model, 'classes_'):
            raise ValueError("Model is not trained! Please re-train your model.")

        return tfidf, model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()


tfidf, model = load_models()
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    return " ".join(
        ps.stem(word) for word in tokens
        if word.isalnum()
        and word not in stopwords.words('english')
    )


# Streamlit UI
st.title("📧 Spam Classifier")
input_sms = st.text_area("Enter message:")

if st.button('Analyze'):
    if not input_sms.strip():
        st.warning("Please enter a message!")
    else:
        try:
            processed = transform_text(input_sms)
            vector = tfidf.transform([processed])

            # Additional verification
            if not hasattr(model, 'predict'):
                raise ValueError("Invalid model - missing predict method")

            prediction = model.predict(vector)[0]
            confidence = model.predict_proba(vector)[0].max()

            if prediction == 1:
                st.error(f"🚨 SPAM (confidence: {confidence:.0%})")
            else:
                st.success(f"✅ HAM (confidence: {confidence:.0%})")

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")