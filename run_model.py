# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')

# Current folder or nested folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NESTED_DIR = os.path.join(CURRENT_DIR, "multilingual_sentiment_model")

if os.path.exists(os.path.join(NESTED_DIR, "model.safetensors")):
    MODEL_DIR = NESTED_DIR
elif os.path.exists(os.path.join(CURRENT_DIR, "model.safetensors")):
    MODEL_DIR = CURRENT_DIR
else:
    raise FileNotFoundError("Could not find model.safetensors in this folder!")

print("=================================================================")
print("  🚀 MULTILINGUAL SENTIMENT CLASSIFIER (English, Tamil, Tanglish)")
print("=================================================================")
print(f"Loading model from: {MODEL_DIR}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

id2label = model.config.id2label

import re

def normalize_tanglish(text):
    # Normalize colloquial spoken Tanglish variations to standard transliterations
    # e.g., 'intha mura' -> 'intha murai', 'irunthiccu' -> 'irundhadhu'
    text = re.sub(r'\b(intha|oru|andha|rendu)\s+mura\b', r'\1 murai', text, flags=re.IGNORECASE)
    text = re.sub(r'\birunth[i|u]ccu\b', 'irundhadhu', text, flags=re.IGNORECASE)
    return text

def predict(comment):
    clean_text = normalize_tanglish(comment)
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
    
    pred_idx = torch.argmax(probs).item()
    label = id2label[pred_idx]
    confidence = probs[pred_idx].item() * 100
    
    breakdown = {id2label[i].capitalize(): f"{probs[i].item()*100:.1f}%" for i in range(len(probs))}
    return label.upper(), confidence, breakdown

if len(sys.argv) > 1:
    input_text = " ".join(sys.argv[1:])
    sentiment, conf, breakdown = predict(input_text)
    print(f"\nText: \"{input_text}\"")
    print(f"👉 Sentiment: {sentiment} ({conf:.1f}% confidence)")
    print(f"   Probabilities: {breakdown}\n")
    sys.exit(0)

print("\nModel is ready! You can type comments in English, Tamil, or Tanglish.")
print("Type 'exit' or press Ctrl+C to quit.\n")

while True:
    try:
        user_input = input("Enter comment: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Exiting. Goodbye!")
            break
            
        sentiment, conf, breakdown = predict(user_input)
        print(f"👉 Sentiment: {sentiment} ({conf:.1f}% confidence)")
        print(f"   Probabilities: {breakdown}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting. Goodbye!")
        break
