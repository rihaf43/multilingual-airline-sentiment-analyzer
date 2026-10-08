# -*- coding: utf-8 -*-
import os
import sys
import threading
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import customtkinter as ctk

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

# ==========================================================
# 🎨 VIBRANT PURPLE CLAY PALETTE (Matching User Reference)
# ==========================================================
CANVAS_BG = "#6C52F8"        # Rich Vibrant Purple Canvas from user image
PURPLE_HERO = "#5D41F5"      # Deep Purple Card
PURPLE_BORDER = "#846EFF"    # Top Highlight Border

WHITE_CARD = "#FFFFFF"       # White Clay Card
WHITE_BORDER = "#F0F2FA"     # Soft bevel border
INPUT_BG = "#F5F7FD"         # Soft inner inset
INPUT_BORDER = "#DCE2F3"

# Primary Action
ACTION_BTN = "#6C52F8"
ACTION_HOVER = "#583CE6"

# Sentiment Colors
POS_BG = "#E8FBF5"
POS_BORDER = "#10B981"
POS_TXT = "#059669"

NEG_BG = "#FEF2F2"
NEG_BORDER = "#EF4444"
NEG_TXT = "#DC2626"

NEU_BG = "#EFF6FF"
NEU_BORDER = "#3B82F6"
NEU_TXT = "#2563EB"

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

class PurpleClayApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Claymorphic Multilingual Sentiment Analyzer")
        self.geometry("780x790")
        self.minsize(720, 720)
        self.configure(fg_color=CANVAS_BG)

        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.create_widgets()
        threading.Thread(target=self.load_model, daemon=True).start()

    def create_widgets(self):
        # Main Scrollable or Padded Container
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=24, pady=20)

        # ----------------- 1. PURPLE HERO CARD -----------------
        hero_card = ctk.CTkFrame(
            container,
            corner_radius=28,
            border_width=3,
            border_color=PURPLE_BORDER,
            fg_color=PURPLE_HERO
        )
        hero_card.pack(fill="x", pady=(0, 14))

        top_row = ctk.CTkFrame(hero_card, fg_color="transparent")
        top_row.pack(fill="x", padx=20, pady=(16, 6))

        title = ctk.CTkLabel(
            top_row,
            text="✨ Claymorphism AI",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="white"
        )
        title.pack(side="left")

        self.badge_tag = ctk.CTkLabel(
            top_row,
            text="3-LANG MODEL",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="white",
            fg_color="#7B64FF",
            corner_radius=12,
            padx=10,
            pady=3
        )
        self.badge_tag.pack(side="right")

        # Floating Wallet/Credit Card Style Sub-Banner
        credit_box = ctk.CTkFrame(hero_card, corner_radius=18, fg_color="#4F32EA")
        credit_box.pack(fill="x", padx=20, pady=(0, 16))

        ctk.CTkLabel(
            credit_box,
            text="SUPPORTED LANGUAGES",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color="#B6A8FF"
        ).pack(anchor="w", padx=16, pady=(10, 0))

        ctk.CTkLabel(
            credit_box,
            text="English  •  தமிழ் (Tamil)  •  Tanglish",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color="white"
        ).pack(anchor="w", padx=16, pady=(0, 10))

        # ----------------- 2. WHITE CLAY WORKSPACE -----------------
        white_card = ctk.CTkFrame(
            container,
            corner_radius=32,
            border_width=3,
            border_color="#FFFFFF",
            fg_color=WHITE_CARD
        )
        white_card.pack(fill="both", expand=True)

        # Quick Pastel Pills Row (Reference Style)
        pill_row = ctk.CTkFrame(white_card, fg_color="transparent")
        pill_row.pack(fill="x", padx=20, pady=(18, 12))

        ctk.CTkLabel(
            pill_row,
            text="Quick Samples:",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#94A3B8"
        ).pack(side="left", padx=(0, 6))

        pills = [
            ("🇬🇧 English", "The flight was amazing, thank you so much!", "#EAFBF5", "#10B981"),
            ("🇮🇳 தமிழ்", "ரொம்ப மோசமான அனுபவம், லக்கேஜ் காணாம போயிடுச்சு", "#FFF1F3", "#FF5A79"),
            ("💬 Tanglish", "intha mura service ellam nalla irunthiccu", "#F3EFFF", "#7952F5"),
            ("❓ Inquiry", "Flight 302 eppo kelambum?", "#EBFBFF", "#00BCD4")
        ]

        for label, txt, bg, border_c in pills:
            btn = ctk.CTkButton(
                pill_row,
                text=label,
                corner_radius=18,
                border_width=2,
                border_color=border_c,
                fg_color=bg,
                hover_color=border_c,
                text_color=border_c,
                font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                height=28,
                width=84,
                command=lambda t=txt: self.insert_sample(t)
            )
            btn.pack(side="left", padx=3)

        # Text Input
        self.text_input = ctk.CTkTextbox(
            white_card,
            height=85,
            corner_radius=20,
            border_width=2,
            border_color=INPUT_BORDER,
            fg_color=INPUT_BG,
            text_color="#1E293B",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            wrap="word"
        )
        self.text_input.pack(fill="x", padx=20, pady=(0, 10))
        self.text_input.bind("<Control-Return>", lambda e: self.on_analyze())

        # Action Buttons
        btn_box = ctk.CTkFrame(white_card, fg_color="transparent")
        btn_box.pack(fill="x", padx=20, pady=(0, 14))

        self.analyze_btn = ctk.CTkButton(
            btn_box,
            text="Analyze Sentiment 🚀",
            corner_radius=24,
            border_width=2,
            border_color="#5136E6",
            fg_color=ACTION_BTN,
            hover_color=ACTION_HOVER,
            text_color="white",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            height=44,
            command=self.on_analyze,
            state="disabled"
        )
        self.analyze_btn.pack(side="left", expand=True, fill="x", padx=(0, 10))

        clear_btn = ctk.CTkButton(
            btn_box,
            text="Clear",
            corner_radius=24,
            border_width=2,
            border_color="#CBD5E1",
            fg_color="#F1F5F9",
            hover_color="#E2E8F0",
            text_color="#64748B",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            height=44,
            width=90,
            command=self.clear_input
        )
        clear_btn.pack(side="right")

        # ----------------- 3. OUTCOME SECTION -----------------
        self.badge_card = ctk.CTkFrame(
            white_card,
            corner_radius=22,
            border_width=2,
            border_color="#E2E8F0",
            fg_color="#F8FAFC",
            height=58
        )
        self.badge_card.pack(fill="x", padx=20, pady=(0, 14))
        self.badge_card.pack_propagate(False)

        self.badge_lbl = ctk.CTkLabel(
            self.badge_card,
            text="Enter a comment above...",
            font=ctk.CTkFont(family="Segoe UI", size=17, weight="bold"),
            text_color="#94A3B8"
        )
        self.badge_lbl.pack(expand=True)

        # 3D Probability Meters (Bar Meters like reference)
        meter_box = ctk.CTkFrame(white_card, fg_color="transparent")
        meter_box.pack(fill="x", padx=22, pady=(0, 16))

        self.pos_lbl, self.pos_bar = self.create_clay_bar(meter_box, "Positive 😊", "#10B981")
        self.neu_lbl, self.neu_bar = self.create_clay_bar(meter_box, "Neutral 😐", "#3B82F6")
        self.neg_lbl, self.neg_bar = self.create_clay_bar(meter_box, "Negative 😡", "#EF4444")

    def create_clay_bar(self, parent, label_text, color):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=4)

        lbl = ctk.CTkLabel(
            row,
            text=f"{label_text}: 0.0%",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#475569",
            width=140,
            anchor="w"
        )
        lbl.pack(side="left")

        bar = ctk.CTkProgressBar(
            row,
            height=16,
            corner_radius=10,
            border_width=2,
            border_color="#E2E8F0",
            progress_color=color,
            fg_color="#F1F5F9"
        )
        bar.set(0.0)
        bar.pack(side="left", expand=True, fill="x", padx=(10, 0))

        return lbl, bar

    def insert_sample(self, text):
        self.text_input.delete("1.0", "end")
        self.text_input.insert("1.0", text)
        self.on_analyze()

    def clear_input(self):
        self.text_input.delete("1.0", "end")
        self.badge_card.configure(fg_color="#F8FAFC", border_color="#E2E8F0")
        self.badge_lbl.configure(text="Enter a comment above...", text_color="#94A3B8")
        self.pos_lbl.configure(text="Positive 😊: 0.0%")
        self.pos_bar.set(0.0)
        self.neu_lbl.configure(text="Neutral 😐: 0.0%")
        self.neu_bar.set(0.0)
        self.neg_lbl.configure(text="Negative 😡: 0.0%")
        self.neg_bar.set(0.0)

    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(self.device)
            self.model.eval()

            self.badge_tag.configure(text="READY ✨", fg_color="#10B981")
            self.analyze_btn.configure(state="normal")
        except Exception as e:
            self.badge_tag.configure(text="ERROR ❌", fg_color="#EF4444")

    def on_analyze(self):
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            return

        self.analyze_btn.configure(state="disabled", text="Analyzing...")
        threading.Thread(target=self.run_inference, args=(text,), daemon=True).start()

    def run_inference(self, text):
        clean_text = re.sub(r'\b(intha|oru|andha|rendu)\s+mura\b', r'\1 murai', text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\birunth[i|u]ccu\b', 'irundhadhu', clean_text, flags=re.IGNORECASE)

        try:
            inputs = self.tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)[0]

            pred_idx = torch.argmax(probs).item()
            label = self.model.config.id2label[pred_idx].lower()
            confidence = probs[pred_idx].item() * 100

            pos_p = probs[2].item()
            neu_p = probs[1].item()
            neg_p = probs[0].item()

            self.after(0, self.update_results, label, confidence, pos_p, neu_p, neg_p)
        finally:
            self.after(0, lambda: self.analyze_btn.configure(state="normal", text="Analyze Sentiment 🚀"))

    def update_results(self, label, confidence, pos_p, neu_p, neg_p):
        if label == "positive":
            bg, border, txt = POS_BG, POS_BORDER, POS_TXT
            badge_text = f"😊 POSITIVE ({confidence:.1f}% confidence)"
        elif label == "negative":
            bg, border, txt = NEG_BG, NEG_BORDER, NEG_TXT
            badge_text = f"😡 NEGATIVE ({confidence:.1f}% confidence)"
        else:
            bg, border, txt = NEU_BG, NEU_BORDER, NEU_TXT
            badge_text = f"😐 NEUTRAL ({confidence:.1f}% confidence)"

        self.badge_card.configure(fg_color=bg, border_color=border)
        self.badge_lbl.configure(text=badge_text, text_color=txt)

        self.pos_lbl.configure(text=f"Positive 😊: {pos_p*100:.1f}%")
        self.pos_bar.set(pos_p)

        self.neu_lbl.configure(text=f"Neutral 😐: {neu_p*100:.1f}%")
        self.neu_bar.set(neu_p)

        self.neg_lbl.configure(text=f"Negative 😡: {neg_p*100:.1f}%")
        self.neg_bar.set(neg_p)

if __name__ == "__main__":
    app = PurpleClayApp()
    app.mainloop()
