import streamlit as st
import pickle
import numpy as np

# ------------------------------
# Load saved model and vectorizer
# ------------------------------
@st.cache_resource
def load_model():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("spam_model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_model()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📩 SMS Spam Detection App")
st.write("Enter a message below to check whether it is **Spam** or **Not Spam**.")

# Input box
sms = st.text_area("✉️ Enter your SMS message:")

# Prediction button
if st.button("🔍 Check Spam"):
    if sms.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        # Transform and predict
        transformed = vectorizer.transform([sms])
        result = model.predict(transformed)[0]

        if result == 1:
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **NOT SPAM**.")

# Footer
st.caption("Built with ❤️ using Streamlit and Scikit-learn")
