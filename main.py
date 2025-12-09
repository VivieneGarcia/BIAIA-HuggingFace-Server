# api/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import os

print("--- Loading Sentiment Model ---")
sentiment_classifier = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base", 
    return_all_scores=True
)
print("--- Model Loaded Successfully ---\n")

EMOJI_MAP = {
    "anger": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "joy": "😃",
    "sadness": "😢",
    "surprise": "😲",
    "neutral": "😐",
}

app = FastAPI()

# allow calls from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev, tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/sentiment")
async def analyze(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"error": "Missing text"}

    results = sentiment_classifier(text)[0]  # returns list of dicts with 'label' and 'score'
    scores = {item['label'].lower(): item['score'] for item in results}

    # fallback for missing emotions
    for key in EMOJI_MAP:
        scores.setdefault(key, 0.0)

    top_emotion = max(scores, key=scores.get)
    return {"sentiment_scores": scores, "sentiment_label": top_emotion}
