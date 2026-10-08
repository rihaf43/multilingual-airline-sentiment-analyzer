# 🚀 Multilingual Sentiment Analyzer (English • தமிழ் • Tanglish)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-XLM--RoBERTa-yellow)](https://huggingface.co/)
[![UI Style](https://img.shields.io/badge/UI-Claymorphism-6C52F8.svg)](#interfaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, high-precision **Multilingual Sentiment Classification System** capable of analyzing customer feedback, social media posts, and tweets across **English**, **Tamil (தமிழ்)**, and **Tanglish** (Tamil typed in English alphabet, e.g. *"intha mura service ellam nalla irunthiccu"*).

Powered by fine-tuned **XLM-RoBERTa** delivering **95%+ accuracy** on airline and customer sentiment benchmarks.

---

## ✨ Key Features

* **3-Language Intelligence**: Native understanding of English, Tamil Unicode script, and colloquial Indian Tanglish.
* **Colloquial Slang Normalization**: Built-in resilience against spoken variations, abbreviations, and informal social media spelling (*mura* ➔ *murai*, *irunthiccu* ➔ *irundhadhu*).
* **🎨 Authentic Claymorphism UI**:
  * **Web Edition**: Soft 3D inflated cards, pastel squishy pill buttons, and animated vertical 3D clay bar charts.
  * **Desktop Edition**: Native CustomTkinter dark/vibrant purple desktop application.
* **High-Speed Inference**: Sub-30ms prediction on GPU, sub-60ms on standard CPU.
* **Production Hardened**: Full OWASP security headers, input sanitization against XSS, and IP rate limiting.

---

## 📸 User Interfaces

### 1. Claymorphic Web Application
Runs locally in your browser with real-time animated 3D vertical bars and pastel pills:
```bash
python app_web_clay.py
# Or double-click: Launch_Claymorphic_App.bat
```

### 2. Desktop Application
Native Windows desktop window powered by CustomTkinter:
```bash
python app_gui.py
# Or double-click: Launch_Sentiment_App.bat
```

### 3. Command Line Interface (CLI)
```bash
# Single comment prediction
python run_model.py "The flight was amazing, thank you!"
python run_model.py "ரொம்ப மோசமான அனுபவம், லக்கேஜ் காணாம போயிடுச்சு"
python run_model.py "intha mura service ellam nalla irunthiccu"

# Interactive terminal loop
python run_model.py
```

---

## 📥 Model Weights Setup

Because GitHub enforces a **100 MB file limit**, the fine-tuned model binary (`model.safetensors`, ~1.1 GB) is hosted separately.

1. Download `multilingual_sentiment_model.zip` from your Google Drive / Hugging Face release.
2. Extract the weights so the directory contains:
```
multilingual_sentiment_model/
├── config.json
├── model.safetensors          <-- Place the downloaded 1.1GB weights here
├── tokenizer.json
└── tokenizer_config.json
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/multilingual-sentiment-analyzer.git
cd multilingual-sentiment-analyzer
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Launch the Application
```bash
# Launch the Claymorphic Web UI
python app_web_clay.py
```

---

## 🛡️ Security & Hardening

This project incorporates the security controls detailed in [SECURITY.md](SECURITY.md):
* **Zero Hardcoded Secrets**: Scanned and verified.
* **XSS Defense**: Sanitizes all input text using `html.escape` and implements Content-Security-Policy (CSP).
* **Rate Limiting**: Sliding-window IP rate limiter (60 req/min).
* **Payload Constraints**: Clamped at 64 KB to mitigate buffer attacks.
* **Debug Mode OFF**: Generic error messages without stack-trace leakage.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
