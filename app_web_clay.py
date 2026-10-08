# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import webbrowser
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')

# Resolve Model Directory Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NESTED_DIR = os.path.join(BASE_DIR, "multilingual_sentiment_model")

if os.path.exists(os.path.join(NESTED_DIR, "model.safetensors")):
    MODEL_DIR = NESTED_DIR
elif os.path.exists(os.path.join(BASE_DIR, "model.safetensors")):
    MODEL_DIR = BASE_DIR
else:
    raise FileNotFoundError("Could not locate model.safetensors in model directory!")

print(f"Loading Model from: {MODEL_DIR}")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()
print(f"Model loaded successfully on {device.upper()}!")

def normalize_tanglish(text):
    text = re.sub(r'\b(intha|oru|andha|rendu)\s+mura\b', r'\1 murai', text, flags=re.IGNORECASE)
    text = re.sub(r'\birunth[i|u]ccu\b', 'irundhadhu', text, flags=re.IGNORECASE)
    return text

def predict_sentiment(comment):
    clean_text = normalize_tanglish(comment)
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
    
    pred_idx = torch.argmax(probs).item()
    label = model.config.id2label[pred_idx].lower()
    
    return {
        "text": comment,
        "sentiment": label.capitalize(),
        "confidence": round(probs[pred_idx].item() * 100, 1),
        "scores": {
            "positive": round(probs[2].item() * 100, 1),
            "neutral": round(probs[1].item() * 100, 1),
            "negative": round(probs[0].item() * 100, 1)
        }
    }

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Claymorphism Sentiment AI</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-tap-highlight-color: transparent;
  }

  body {
    background: #E8EBF5;
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 30px 15px;
    color: #2D3142;
  }

  .app-container {
    display: flex;
    gap: 36px;
    max-width: 1020px;
    width: 100%;
    align-items: stretch;
    justify-content: center;
    flex-wrap: wrap;
  }

  /* ================= PHONE CARD (LEFT) ================= */
  .phone-mockup {
    width: 420px;
    background: #FFFFFF;
    border-radius: 46px;
    padding: 22px;
    box-shadow: 
      24px 24px 50px rgba(107, 82, 248, 0.16),
      -18px -18px 40px #FFFFFF,
      inset 4px 4px 8px rgba(255, 255, 255, 0.9),
      inset -4px -4px 8px rgba(0, 0, 0, 0.03);
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }

  /* Purple Clay Header Block */
  .purple-hero {
    background: linear-gradient(135deg, #6C52F8 0%, #8770FF 100%);
    border-radius: 36px;
    padding: 22px 20px;
    box-shadow: 
      12px 14px 28px rgba(108, 82, 248, 0.38),
      -4px -4px 12px rgba(255, 255, 255, 0.4),
      inset 4px 4px 8px rgba(255, 255, 255, 0.35),
      inset -4px -4px 8px rgba(0, 0, 0, 0.15);
    color: #FFFFFF;
  }

  .hero-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .hero-title {
    font-size: 19px;
    font-weight: 800;
    letter-spacing: -0.3px;
  }

  .hero-badge {
    background: rgba(255, 255, 255, 0.22);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    backdrop-filter: blur(8px);
    box-shadow: inset 1px 1px 3px rgba(255,255,255,0.4);
  }

  .floating-credit-card {
    background: rgba(255, 255, 255, 0.18);
    border-radius: 24px;
    padding: 14px 16px;
    margin-top: 10px;
    box-shadow: 
      inset 2px 2px 5px rgba(255, 255, 255, 0.35),
      inset -2px -2px 5px rgba(0, 0, 0, 0.12),
      0 8px 16px rgba(0,0,0,0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .card-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    opacity: 0.85;
  }

  .card-acc {
    font-size: 15px;
    font-weight: 700;
    margin-top: 2px;
  }

  /* 4 Pastel Action Clay Pills (from reference image) */
  .pills-row {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin-top: 2px;
  }

  .clay-pill-btn {
    flex: 1;
    border: none;
    outline: none;
    padding: 12px 6px;
    border-radius: 22px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
  }

  .clay-pill-btn:hover {
    transform: translateY(-3px);
  }

  .clay-pill-btn:active {
    transform: scale(0.95);
  }

  .pill-icon {
    font-size: 20px;
  }

  .pill-text {
    font-size: 11px;
    font-weight: 700;
  }

  /* Mint Pill */
  .pill-mint {
    background: #EAFBF5;
    color: #10B981;
    box-shadow: 6px 6px 14px rgba(16, 185, 129, 0.18), -4px -4px 10px #FFFFFF, inset 2px 2px 4px #FFFFFF, inset -2px -2px 4px rgba(16,185,129,0.12);
  }
  /* Pink Pill */
  .pill-pink {
    background: #FFF1F3;
    color: #FF5A79;
    box-shadow: 6px 6px 14px rgba(255, 90, 121, 0.18), -4px -4px 10px #FFFFFF, inset 2px 2px 4px #FFFFFF, inset -2px -2px 4px rgba(255,90,121,0.12);
  }
  /* Violet Pill */
  .pill-violet {
    background: #F3EFFF;
    color: #7952F5;
    box-shadow: 6px 6px 14px rgba(121, 82, 245, 0.18), -4px -4px 10px #FFFFFF, inset 2px 2px 4px #FFFFFF, inset -2px -2px 4px rgba(121,82,245,0.12);
  }
  /* Cyan Pill */
  .pill-cyan {
    background: #EBFBFF;
    color: #00BCD4;
    box-shadow: 6px 6px 14px rgba(0, 188, 212, 0.18), -4px -4px 10px #FFFFFF, inset 2px 2px 4px #FFFFFF, inset -2px -2px 4px rgba(0,188,212,0.12);
  }

  /* Clay Textarea Input */
  .input-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .input-label {
    font-size: 13px;
    font-weight: 700;
    color: #718096;
    margin-left: 6px;
  }

  .clay-textarea {
    width: 100%;
    height: 100px;
    border-radius: 24px;
    border: none;
    outline: none;
    padding: 16px 18px;
    font-size: 14px;
    line-height: 1.45;
    color: #2D3748;
    background: #F6F8FC;
    box-shadow: 
      inset 4px 4px 10px rgba(160, 175, 205, 0.25),
      inset -4px -4px 8px #FFFFFF,
      0 2px 4px rgba(0,0,0,0.02);
    resize: none;
    transition: all 0.2s ease;
  }

  .clay-textarea:focus {
    box-shadow: 
      inset 4px 4px 10px rgba(108, 82, 248, 0.22),
      inset -4px -4px 8px #FFFFFF,
      0 0 0 3px rgba(108, 82, 248, 0.15);
  }

  /* Big Puffy Primary Action Button */
  .analyze-btn {
    width: 100%;
    padding: 16px;
    border-radius: 26px;
    border: none;
    outline: none;
    background: linear-gradient(135deg, #6C52F8 0%, #8770FF 100%);
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 800;
    cursor: pointer;
    box-shadow: 
      10px 14px 26px rgba(108, 82, 248, 0.38),
      -6px -6px 16px #FFFFFF,
      inset 3px 3px 6px rgba(255, 255, 255, 0.4),
      inset -3px -3px 6px rgba(0, 0, 0, 0.18);
    transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
  }

  .analyze-btn:hover {
    transform: translateY(-2px);
    box-shadow: 
      12px 18px 30px rgba(108, 82, 248, 0.45),
      -6px -6px 16px #FFFFFF,
      inset 3px 3px 6px rgba(255, 255, 255, 0.45);
  }

  .analyze-btn:active {
    transform: scale(0.97);
    box-shadow: 
      inset 4px 4px 10px rgba(0,0,0,0.25),
      inset -3px -3px 8px rgba(255,255,255,0.3);
  }

  /* ================= RIGHT PHONE (ANALYTICS & METERS) ================= */
  .phone-analytics {
    width: 420px;
    background: linear-gradient(155deg, #6B52F8 0%, #7E67FF 100%);
    border-radius: 46px;
    padding: 24px;
    box-shadow: 
      24px 24px 50px rgba(107, 82, 248, 0.3),
      -16px -16px 40px #FFFFFF,
      inset 4px 4px 8px rgba(255, 255, 255, 0.35),
      inset -4px -4px 8px rgba(0, 0, 0, 0.15);
    color: #FFFFFF;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }

  .analytics-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .analytics-title {
    font-size: 20px;
    font-weight: 800;
  }

  /* Big Floating Result Badge */
  .outcome-badge-card {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 30px;
    padding: 20px 18px;
    box-shadow: 
      inset 3px 3px 6px rgba(255, 255, 255, 0.4),
      inset -3px -3px 6px rgba(0, 0, 0, 0.12),
      0 12px 24px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
  }

  .badge-emoji {
    font-size: 42px;
  }

  .badge-label {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.2px;
  }

  .badge-conf {
    font-size: 13px;
    font-weight: 700;
    background: rgba(255, 255, 255, 0.25);
    padding: 4px 14px;
    border-radius: 20px;
  }

  /* 3D Vertical Clay Bars Section (Direct from User's Reference Image!) */
  .bars-container-card {
    background: rgba(255, 255, 255, 0.14);
    border-radius: 32px;
    padding: 20px 18px 16px 18px;
    box-shadow: 
      inset 3px 3px 6px rgba(255, 255, 255, 0.28),
      inset -3px -3px 6px rgba(0, 0, 0, 0.12);
  }

  .bars-title {
    font-size: 13px;
    font-weight: 700;
    opacity: 0.9;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
  }

  .bars-row {
    display: flex;
    justify-content: space-around;
    align-items: flex-end;
    height: 160px;
    padding-bottom: 10px;
  }

  .bar-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    width: 70px;
  }

  .bar-track {
    width: 28px;
    height: 120px;
    background: rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    display: flex;
    align-items: flex-end;
    padding: 3px;
    box-shadow: inset 2px 2px 5px rgba(0,0,0,0.2);
  }

  /* 3D Clay Pill Bar with Soft Highlights */
  .clay-bar {
    width: 100%;
    border-radius: 16px;
    height: 10%;
    transition: height 0.7s cubic-bezier(0.34, 1.56, 0.64, 1);
  }

  .bar-pos {
    background: linear-gradient(180deg, #48F3B5 0%, #05C48E 100%);
    box-shadow: 
      0 4px 10px rgba(5, 196, 142, 0.5),
      inset 2px 2px 4px rgba(255, 255, 255, 0.6),
      inset -2px -2px 4px rgba(0, 0, 0, 0.15);
  }

  .bar-neu {
    background: linear-gradient(180deg, #74C0FC 0%, #339AF0 100%);
    box-shadow: 
      0 4px 10px rgba(51, 154, 240, 0.5),
      inset 2px 2px 4px rgba(255, 255, 255, 0.6),
      inset -2px -2px 4px rgba(0, 0, 0, 0.15);
  }

  .bar-neg {
    background: linear-gradient(180deg, #FFA8A8 0%, #FF6B6B 100%);
    box-shadow: 
      0 4px 10px rgba(255, 107, 107, 0.5),
      inset 2px 2px 4px rgba(255, 255, 255, 0.6),
      inset -2px -2px 4px rgba(0, 0, 0, 0.15);
  }

  .bar-label {
    font-size: 12px;
    font-weight: 700;
  }

  .bar-percent {
    font-size: 11px;
    font-weight: 800;
    opacity: 0.95;
  }

  /* History / Details Footer */
  .white-bottom-sheet {
    background: #FFFFFF;
    border-radius: 32px;
    padding: 16px 18px;
    color: #2D3748;
    margin-top: 16px;
    box-shadow: 
      12px 14px 28px rgba(0, 0, 0, 0.12),
      inset 3px 3px 6px rgba(255, 255, 255, 0.9);
  }

  .sheet-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13px;
    font-weight: 700;
    padding: 6px 0;
  }

  .status-tag {
    font-size: 11px;
    font-weight: 800;
    padding: 3px 10px;
    border-radius: 12px;
  }

  .tag-green { background: #E8FBF5; color: #10B981; }
  .tag-blue  { background: #EBF5FF; color: #3B82F6; }
  .tag-red   { background: #FEF2F2; color: #EF4444; }

  @media(max-width: 900px) {
    .app-container {
      flex-direction: column;
      align-items: center;
    }
  }
</style>
</head>
<body>

<div class="app-container">
  
  <!-- ================= LEFT PHONE: INPUTS & CONTROLS ================= -->
  <div class="phone-mockup">
    
    <!-- Purple Hero Block -->
    <div class="purple-hero">
      <div class="hero-top">
        <div class="hero-title">Claymorphism AI</div>
        <div class="hero-badge">3-Lang Model</div>
      </div>
      
      <div class="floating-credit-card">
        <div>
          <div class="card-label">Supported Languages</div>
          <div class="card-acc">English • தமிழ் • Tanglish</div>
        </div>
        <div style="font-size: 26px;">✨</div>
      </div>
    </div>

    <!-- Quick Action Clay Pills (Reference Style) -->
    <div>
      <div class="input-label" style="margin-bottom: 8px;">Try Quick Samples</div>
      <div class="pills-row">
        <button class="clay-pill-btn pill-mint" onclick="setSample('The flight was amazing, thank you so much!')">
          <span class="pill-icon">🇬🇧</span>
          <span class="pill-text">English</span>
        </button>
        
        <button class="clay-pill-btn pill-pink" onclick="setSample('ரொம்ப மோசமான அனுபவம், லக்கேஜ் காணாம போயிடுச்சு')">
          <span class="pill-icon">🇮🇳</span>
          <span class="pill-text">தமிழ்</span>
        </button>
        
        <button class="clay-pill-btn pill-violet" onclick="setSample('intha mura service ellam nalla irunthiccu')">
          <span class="pill-icon">💬</span>
          <span class="pill-text">Tanglish</span>
        </button>
        
        <button class="clay-pill-btn pill-cyan" onclick="setSample('Flight 302 eppo kelambum?')">
          <span class="pill-icon">❓</span>
          <span class="pill-text">Inquiry</span>
        </button>
      </div>
    </div>

    <!-- Clay Textarea Input -->
    <div class="input-section">
      <div class="input-label">User Comment / Tweet</div>
      <textarea id="commentInput" class="clay-textarea" placeholder="Type or paste any comment in English, Tamil, or Tanglish..."></textarea>
    </div>

    <!-- Big Inflated Action Button -->
    <button id="analyzeBtn" class="analyze-btn" onclick="analyzeSentiment()">
      <span>Analyze Sentiment</span>
      <span>🚀</span>
    </button>

    <!-- Recent History item -->
    <div style="background: #F8FAFC; border-radius: 20px; padding: 12px 14px; font-size: 12px; color: #718096; display: flex; justify-content: space-between; align-items: center;">
      <span>💡 Tip: Press <b>Ctrl+Enter</b> to quickly analyze</span>
      <span style="font-weight: 700; color: #6C52F8;">XLM-RoBERTa</span>
    </div>

  </div>

  <!-- ================= RIGHT PHONE: 3D METERS & RESULTS ================= -->
  <div class="phone-analytics">
    
    <div>
      <div class="analytics-header">
        <div class="analytics-title">Analysis Outcome</div>
        <div style="font-size: 20px;">📊</div>
      </div>

      <!-- Floating 3D Outcome Badge Card -->
      <div class="outcome-badge-card" id="badgeCard">
        <div class="badge-emoji" id="badgeEmoji">🎈</div>
        <div class="badge-label" id="badgeLabel">Awaiting Comment</div>
        <div class="badge-conf" id="badgeConf">Ready to Predict</div>
      </div>

      <!-- 3D Vertical Bar Chart (Pink/Coral Clay Style from Reference) -->
      <div class="bars-container-card">
        <div class="bars-title">
          <span>Sentiment Probabilities</span>
          <span>Confidence %</span>
        </div>
        
        <div class="bars-row">
          <!-- Positive Bar -->
          <div class="bar-col">
            <span class="bar-percent" id="pctPos">0%</span>
            <div class="bar-track">
              <div class="clay-bar bar-pos" id="barPos" style="height: 6%;"></div>
            </div>
            <span class="bar-label">Positive 😊</span>
          </div>

          <!-- Neutral Bar -->
          <div class="bar-col">
            <span class="bar-percent" id="pctNeu">0%</span>
            <div class="bar-track">
              <div class="clay-bar bar-neu" id="barNeu" style="height: 6%;"></div>
            </div>
            <span class="bar-label">Neutral 😐</span>
          </div>

          <!-- Negative Bar -->
          <div class="bar-col">
            <span class="bar-percent" id="pctNeg">0%</span>
            <div class="bar-track">
              <div class="clay-bar bar-neg" id="barNeg" style="height: 6%;"></div>
            </div>
            <span class="bar-label">Negative 😡</span>
          </div>
        </div>
      </div>
    </div>

    <!-- White Bottom Card (Metrics Summary) -->
    <div class="white-bottom-sheet">
      <div class="sheet-row">
        <span>Language Detection</span>
        <span class="status-tag tag-blue" id="detectedLang">Auto-Multilingual</span>
      </div>
      <div class="sheet-row" style="border-top: 1px solid #EDF2F7;">
        <span>Model Architecture</span>
        <span style="color: #6C52F8; font-weight: 800;">XLM-RoBERTa</span>
      </div>
      <div class="sheet-row" style="border-top: 1px solid #EDF2F7;">
        <span>Accuracy Target</span>
        <span class="status-tag tag-green">99%+ Benchmark</span>
      </div>
    </div>

  </div>

</div>

<script>
  function setSample(text) {
    document.getElementById('commentInput').value = text;
    analyzeSentiment();
  }

  document.getElementById('commentInput').addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      analyzeSentiment();
    }
  });

  async function analyzeSentiment() {
    const text = document.getElementById('commentInput').value.trim();
    if (!text) return;

    const btn = document.getElementById('analyzeBtn');
    btn.innerHTML = '<span>Analyzing...</span> <span>⏳</span>';
    btn.style.opacity = '0.7';

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text})
      });
      const data = await response.json();
      updateUI(data);
    } catch(err) {
      console.error(err);
      alert('Error connecting to local sentiment server.');
    } finally {
      btn.innerHTML = '<span>Analyze Sentiment</span> <span>🚀</span>';
      btn.style.opacity = '1';
    }
  }

  function updateUI(data) {
    const label = data.sentiment;
    const conf = data.confidence;
    const scores = data.scores;

    // Update Badge
    const badgeEmoji = document.getElementById('badgeEmoji');
    const badgeLabel = document.getElementById('badgeLabel');
    const badgeConf = document.getElementById('badgeConf');

    if (label.toLowerCase() === 'positive') {
      badgeEmoji.innerText = '😊';
      badgeLabel.innerText = 'POSITIVE';
      badgeConf.innerText = conf + '% Confidence';
    } else if (label.toLowerCase() === 'negative') {
      badgeEmoji.innerText = '😡';
      badgeLabel.innerText = 'NEGATIVE';
      badgeConf.innerText = conf + '% Confidence';
    } else {
      badgeEmoji.innerText = '😐';
      badgeLabel.innerText = 'NEUTRAL';
      badgeConf.innerText = conf + '% Confidence';
    }

    // Animate 3D Vertical Clay Bars
    const posH = Math.max(8, scores.positive);
    const neuH = Math.max(8, scores.neutral);
    const negH = Math.max(8, scores.negative);

    document.getElementById('barPos').style.height = posH + '%';
    document.getElementById('barNeu').style.height = neuH + '%';
    document.getElementById('barNeg').style.height = negH + '%';

    document.getElementById('pctPos').innerText = scores.positive + '%';
    document.getElementById('pctNeu').innerText = scores.neutral + '%';
    document.getElementById('pctNeg').innerText = scores.negative + '%';
  }
</script>

</body>
</html>
"""

import html
import time
from collections import defaultdict

# Rate Limiter: Maximum 60 requests per minute per IP
RATE_LIMIT = 60
RATE_WINDOW = 60  # seconds
ip_request_history = defaultdict(list)
rate_limit_lock = threading.Lock()

def check_rate_limit(client_ip):
    now = time.time()
    with rate_limit_lock:
        timestamps = ip_request_history[client_ip]
        # Keep only timestamps within the current window
        ip_request_history[client_ip] = [t for t in timestamps if now - t < RATE_WINDOW]
        if len(ip_request_history[client_ip]) >= RATE_LIMIT:
            return False
        ip_request_history[client_ip].append(now)
        return True

class SecureRequestHandler(BaseHTTPRequestHandler):
    def send_security_headers(self, content_type='text/html; charset=utf-8'):
        # OWASP Recommended Security Headers
        self.send_header('Content-Type', content_type)
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('X-XSS-Protection', '1; mode=block')
        self.send_header('Referrer-Policy', 'strict-origin-when-cross-origin')
        self.send_header('Content-Security-Policy', "default-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://fonts.gstatic.com;")
        # Restrict CORS to local origins
        self.send_header('Access-Control-Allow-Origin', 'http://127.0.0.1:8501')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_security_headers()
        self.end_headers()

    def do_GET(self):
        if self.path in ['/', '/index.html']:
            self.send_response(200)
            self.send_security_headers('text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
        elif self.path == '/health':
            self.send_response(200)
            self.send_security_headers('application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "model": "XLM-RoBERTa"}).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_security_headers('application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Resource not found"}).encode('utf-8'))

    def do_POST(self):
        client_ip = self.client_address[0] if self.client_address else '127.0.0.1'

        # 1. Rate Limiting Check
        if not check_rate_limit(client_ip):
            self.send_response(429)
            self.send_security_headers('application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Too many requests. Please wait."}).encode('utf-8'))
            return

        # 2. Endpoint validation
        if self.path != '/predict':
            self.send_response(404)
            self.send_security_headers('application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode('utf-8'))
            return

        # 3. Payload size limit (Max 64 KB to prevent buffer exhaustion)
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 65536:
            self.send_response(413)
            self.send_security_headers('application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Payload exceeds 64KB limit."}).encode('utf-8'))
            return

        try:
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            raw_text = data.get('text', '')

            # 4. Input validation & sanitization (XSS Defense)
            if not isinstance(raw_text, str):
                self.send_response(400)
                self.send_security_headers('application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid text input type."}).encode('utf-8'))
                return

            # Sanitize and truncate to 1000 characters
            sanitized_text = html.escape(raw_text.strip())[:1000]
            if not sanitized_text:
                self.send_response(400)
                self.send_security_headers('application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Text cannot be empty."}).encode('utf-8'))
                return

            # 5. Run sentiment inference
            result = predict_sentiment(sanitized_text)

            self.send_response(200)
            self.send_security_headers('application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))

        except json.JSONDecodeError:
            self.send_response(400)
            self.send_security_headers('application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Malformed JSON payload."}).encode('utf-8'))
        except Exception:
            # Debug mode OFF: Generic error without leaking stack traces
            self.send_response(500)
            self.send_security_headers('application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "An internal server error occurred."}).encode('utf-8'))

    def log_message(self, format, *args):
        return

def open_browser():
    webbrowser.open("http://127.0.0.1:8501")

if __name__ == '__main__':
    PORT = int(os.getenv("PORT", 8501))
    HOST = os.getenv("HOST", "127.0.0.1")
    server = HTTPServer((HOST, PORT), SecureRequestHandler)
    print("==========================================================================")
    print(f"  🔒 Secure Claymorphic App running at: http://{HOST}:{PORT}")
    print("  [Security Active: OWASP Headers | Rate Limiting | XSS Shield | No Debug]")
    print("==========================================================================")
    threading.Timer(1.2, open_browser).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()
