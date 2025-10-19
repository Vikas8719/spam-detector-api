import streamlit as st
import pickle

# -----------------------------
# Load Model and Vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("spam_model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_model()

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("📩 SMS Spam Detector")
st.write("Enter an SMS message below to check if it is **Spam** or **Not Spam**.")

# User input
user_input = st.text_area("✉️ Message:", height=150)

if st.button("Check Spam"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        # Preprocess and predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        if prediction == 1:
            st.error("🚫 This message is **SPAM**!")
        else:
            st.success("✅ This message is **NOT SPAM**!")

st.markdown("---")
st.caption("Developed by Vikas | Powered by Streamlit & Scikit-learn")
