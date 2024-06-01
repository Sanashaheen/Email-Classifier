import string
import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # lowercase done
    text = nltk.word_tokenize(text)  # 2nd step done tokenization
    y = []
    for i in text:
        if i.isalnum():  # removing special character and append into y list
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # removing stop words and punctuation
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))  # stemming removing
    return " ".join(y)

# Define tfidf variable outside of try block
tfidf = None

try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure that vectorizer.pkl and model.pkl exist.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the Message")

if st.button("Predict"):
    # Ensure tfidf is defined before using it
    if tfidf is None:
        st.error("Error: TF-IDF vectorizer not loaded.")
    else:
        # preprocess
        transformed_sms = transform_text(input_sms)
        # vectorize
        vector_input = tfidf.transform([transformed_sms])
        # predict
        result = model.predict(vector_input)[0]
        # display result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
