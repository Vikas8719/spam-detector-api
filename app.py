from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

# Load model & vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="SMS Spam Detection API")

class Message(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Welcome to the SMS Spam Detection API!"}

@app.post("/predict")
def predict(data: Message):
    text_vector = vectorizer.transform([data.text])
    prediction = model.predict(text_vector)[0]
    result = "Spam" if prediction == 1 else "Ham (Not Spam)"
    return {"prediction": result}
