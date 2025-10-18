import streamlit as st
import pickle
import string
import re

# ------------------------------
# Load the model and vectorizer
# ------------------------------
@st.cache_resource
def load_model():
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    model = pickle.load(open('spam_model.pkl', 'rb'))
    return vectorizer, model

vectorizer, model = load_model()

# ------------------------------
# Text preprocessing function
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    return text

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="📩 SMS Spam Classifier", page_icon="📱")

st.title("📩 SMS Spam Detection App")
st.write("This app uses a trained ML model to detect whether a message is **Spam** or **Not Spam**.")

# User input
message = st.text_area("Enter your SMS message here:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message before predicting.")
    else:
        cleaned_msg = clean_text(message)
        vectorized_msg = vectorizer.transform([cleaned_msg])
        prediction = model.predict(vectorized_msg)[0]

        if prediction == 1:
            st.error("🚨 This message is **Spam!**")
        else:
            st.success("✅ This message is **Not Spam.**")
