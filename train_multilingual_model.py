# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import classification_report, accuracy_score
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Load the 3 cleaned datasets
print("Loading datasets...")
eng = pd.read_csv(r"C:\Users\moham\Downloads\Tweets_cleaned.csv", encoding='utf-8-sig')
tam = pd.read_csv(r"C:\Users\moham\Downloads\tamil_tweets_cleaned.csv", encoding='utf-8-sig')
tg = pd.read_csv(r"C:\Users\moham\Downloads\tanglish_tweets_cleaned.csv", encoding='utf-8-sig')

eng['lang'] = 'en'
tam['lang'] = 'ta'
tg['lang'] = 'tg'

# 2. Combine all 3 datasets
combined = pd.concat([eng, tam, tg], ignore_index=True)
print(f"Total training pool: {len(combined)} rows across English, Tamil, and Tanglish.")

# Split by unique tweet_id so the same original tweet's translations don't leak into test set
unique_ids = eng['tweet_id'].unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=0.15, random_state=42)

train_df = combined[combined['tweet_id'].isin(train_ids)].reset_index(drop=True)
test_df = combined[combined['tweet_id'].isin(test_ids)].reset_index(drop=True)

print(f"Train size: {len(train_df)} rows | Test size: {len(test_df)} rows")

# 3. Build Multilingual Hybrid Feature Union
# Word n-grams capture full vocabulary and idioms
# Char n-grams (char_wb 2-5) capture code-mixing, typos, suffixes (e.g. flight-la, super-a, irundhadhu)
print("\nBuilding Multilingual TF-IDF Feature Union...")
vectorizer = FeatureUnion([
    ('word_tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True)),
    ('char_tfidf', TfidfVectorizer(ngram_range=(2, 5), analyzer='char_wb', max_features=50000, sublinear_tf=True))
])

# LogisticRegression with balanced weights and optimized regularization
clf = LogisticRegression(C=2.5, max_iter=600, class_weight='balanced', solver='lbfgs')

model = Pipeline([
    ('features', vectorizer),
    ('classifier', clf)
])

# 4. Train
start_time = time.time()
print("Training multilingual classifier...")
model.fit(train_df['text'], train_df['airline_sentiment'])
train_time = time.time() - start_time
print(f"Training completed in {train_time:.2f} seconds!")

# 5. Evaluate
preds = model.predict(test_df['text'])
acc = accuracy_score(test_df['airline_sentiment'], preds)
print(f"\n==========================================")
print(f"OVERALL ACCURACY: {acc * 100:.2f}%")
print(f"==========================================")

print("\nClassification Report:")
print(classification_report(test_df['airline_sentiment'], preds))

# Per-language breakdown
test_df['pred'] = preds
print("Accuracy by Language:")
for lang, name in [('en', 'English'), ('ta', 'Tamil'), ('tg', 'Tanglish')]:
    sub = test_df[test_df['lang'] == lang]
    lang_acc = accuracy_score(sub['airline_sentiment'], sub['pred'])
    print(f"  - {name}: {lang_acc * 100:.2f}% ({len(sub)} test samples)")

# 6. Live Test with real multi-lingual comments
test_comments = [
    # English
    ("The flight service was awesome, thank you!", "positive"),
    ("Worst customer support ever, my luggage is lost.", "negative"),
    ("Flight 405 was delayed, what is the new schedule?", "neutral"),
    # Tamil
    ("ரொம்ப அருமையான சேவை, ரொம்ப நன்றி!", "positive"),
    ("ரொம்ப மோசமான அனுபவம், லக்கேஜ் காணாம போயிடுச்சு.", "negative"),
    ("பிளைட் எப்போ வரும்?", "neutral"),
    # Tanglish
    ("Super flight service pa, romba nandri!", "positive"),
    ("Romba worst experience, flight cancel aayiduchu, kaasu refund pannunga!", "negative"),
    ("Flight 302 eppo kelambum?", "neutral"),
    # Mixed / Social media style
    ("Flight delay aanaalum staff romba helpful-a irundhaanga ❤️", "positive"),
    ("Worst service da.. 4 hours wait panna vechiteenga 😡", "negative")
]

print("\n--- LIVE TEST PREDICTIONS ---")
for comment, expected in test_comments:
    pred = model.predict([comment])[0]
    probs = model.predict_proba([comment])[0]
    classes = list(model.classes_)
    prob_dict = {c: f"{p*100:.1f}%" for c, p in zip(classes, probs)}
    conf = max(probs) * 100
    status = "OK" if pred == expected else "MISMATCH"
    print(f"[{pred.upper()}] ({conf:.1f}%) | Expected: {expected} | \"{comment}\"")
