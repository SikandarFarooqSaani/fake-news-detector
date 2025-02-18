import streamlit as st
import torch
import pickle
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import os
import gdown  # If not installed, add to requirements.txt

# URL of the model on Google Drive
model_url = "https://drive.google.com/file/d/1VJFepl6Geno_BlmD5yfrqM9DAidzVLK1/view?usp=sharing"
output_path = "saved_model/model.safetensors"

# Download the model if not present
if not os.path.exists(output_path):
    gdown.download(model_url, output_path, quiet=False)


# Load the model and tokenizer
# Load the model and tokenizer
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

model = GPT2ForSequenceClassification.from_pretrained('saved_model')
tokenizer = GPT2Tokenizer.from_pretrained('saved_model')
model.config.pad_token_id = tokenizer.pad_token_id


# Function to make prediction
def predict_news(news_text):
    inputs = tokenizer(news_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return "Real" if prediction == 1 else "Fake"

# Streamlit App
st.title("Fake News Detection")
st.write("Enter news text to check if it's Fake or Real.")

news_input = st.text_area("News Text", height=200)

if st.button("Predict"):
    if news_input:
        prediction = predict_news(news_input)
        st.subheader("Prediction:")
        if prediction == "Real":
            st.success("This news is likely Real.")
        else:
            st.error("This news is likely Fake.")
    else:
        st.warning("Please enter news text to get a prediction.")
