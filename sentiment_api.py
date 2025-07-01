from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
import uvicorn

# Load the RoBERTa model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define FastAPI app
app = FastAPI(title="RoBERTa Sentiment Analysis API")

# Define input schema
class TextInput(BaseModel):
    text: str

# Load labels (0: negative, 1: neutral, 2: positive)
labels = ['negative', 'neutral', 'positive']

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    pred_label = torch.argmax(probs).item()
    return {
        "label": labels[pred_label],
        "confidence": round(probs[0][pred_label].item(), 4)
    }

@app.post("/analyze")
def analyze_sentiment(input: TextInput):
    return classify_sentiment(input.text)

# Run the server when executed directly
if __name__ == "__main__":
    uvicorn.run("sentiment_api:app", host="127.0.0.1", port=8000, reload=True)
