import streamlit as st
import pickle
import string
import re
import os 
import sys # sys module added for better error logging

# ------------------------------
# Load the model and vectorizer
# ------------------------------
@st.cache_resource
def load_model():
    # Define file paths. Assuming files are in the root directory.
    vectorizer_path = 'tfidf_vectorizer.pkl'
    model_path = 'spam_model.pkl'

    # Check for file existence (Crucial for deployment)
    if not os.path.exists(vectorizer_path):
        st.error(f"❌ Error: Vectorizer file not found at: `{vectorizer_path}`. Check file name case on GitHub.")
        return None, None
    if not os.path.exists(model_path):
        st.error(f"❌ Error: Model file not found at: `{model_path}`. Check file name case on GitHub.")
        return None, None

    try:
        # Load the files
        # We explicitly set encoding="latin1" for pickle to handle different system encodings
        # and prevent potential UnpicklingErrors due to Scikit-learn version mismatches.
        with open(vectorizer_path, 'rb') as f:
             vectorizer = pickle.load(f, encoding='latin1')
        with open(model_path, 'rb') as f:
             model = pickle.load(f, encoding='latin1')
             
        return vectorizer, model
        
    except Exception as e:
        # Log the detailed error to the Streamlit console for user debugging
        print(f"FATAL ERROR DURING MODEL LOADING: {e}", file=sys.stderr)
        st.error(
            "❌ **Critical Error Loading ML Model:** The app failed to load the model or vectorizer. "
            "This is often due to **Scikit-learn version mismatch** between your local machine and the cloud. "
            "Please check the full logs for a detailed traceback (UnpicklingError)."
        )
        return None, None

vectorizer, model = load_model()

# ------------------------------
# Text preprocessing function
# ------------------------------
def clean_text(text):
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', '', text)
    return text

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="📩 SMS Spam Classifier", page_icon="📱", layout="centered")

# Custom CSS for aesthetics (as done previously)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main-header {
        color: #3b82f6; 
        font-size: 2.5em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #10b981; 
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #059669; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-header">📱 SMS Spam Detection App 📩</div>', unsafe_allow_html=True)
st.write("This app uses a trained Machine Learning model (likely **Naive Bayes** or **Logistic Regression**) to detect whether a message is **Spam** or **Not Spam**.")

# User input
message = st.text_area("Enter your SMS message here:", height=150, placeholder="Example: WINNER! You have won a free iPhone. Click the link to claim.")

# Ensure the model loaded successfully before proceeding
if vectorizer is not None and model is not None:
    if st.button("Predict"):
        if message.strip() == "":
            st.warning("Please enter a message before predicting.")
        else:
            # 1. Preprocess
            cleaned_msg = clean_text(message)

            # 2. Vectorize
            vectorized_msg = vectorizer.transform([cleaned_msg])

            # 3. Predict
            prediction_proba = model.predict_proba(vectorized_msg)[0]
            spam_prob = prediction_proba[1] 
            prediction = model.predict(vectorized_msg)[0]

            st.write("---")

            if prediction == 1:
                st.error(f"🚨 This message is **Spam!** (Confidence: {spam_prob:.2%})")
                st.balloons()
            else:
                st.success(f"✅ This message is **Not Spam.** (Confidence: {1 - spam_prob:.2%})")
                st.snow()
else:
    # If model loading failed, show this message instead of hanging the app
    st.markdown("---")
    st.warning("⚠️ **Model initialization failed.** Please check the logs above for file or version errors.")
