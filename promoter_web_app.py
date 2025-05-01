import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# === PAGE CONFIG ===
st.set_page_config(page_title="σ⁵⁴ Promoter Predictor", page_icon="🧬", layout="centered")

# === Load Model & Promoter DB ===
model = load_model("cnn_model.h5")
promoter_df = pd.read_csv("promoter.csv")
promoter_df["PromoterSequence"] = promoter_df["PromoterSequence"].astype(str).str.upper()

# === One-hot Encode DNA ===
def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0],
               'T': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'C': [0, 0, 0, 1]}
    return [mapping.get(base.upper(), [0, 0, 0, 0]) for base in seq]

# === Predict Promoters ===
def predict_sequence(seq):
    predictions = []
    for i in range(len(seq) - 80):
        window = seq[i:i+81]
        encoded = np.array(one_hot_encode(window)).reshape(1, 81, 4)
        prob = model.predict(encoded, verbose=0)[0][0]
        label = 1 if prob > 0.5 else 0
        if label == 1:
            is_known = window in promoter_df["PromoterSequence"].values
            status = "Known Promoter" if is_known else "Unknown Promoter"
            predictions.append((window, prob, status))
    return predictions

# === UI HEADER ===
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧬 σ⁵⁴ Promoter Multi-Sequence Predictor</h1>", unsafe_allow_html=True)

# === About Section ===
st.markdown("""
<div style='
    background-color: #f0f9ff;
    border-left: 6px solid #2196f3;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 25px;
    font-family: "Helvetica", sans-serif;
'>
    <h4 style='color:#0b5394;'>📘 About This Tool</h4>
    <p style='color:#333; font-size:16px; line-height:1.6;'>
    This tool uses a trained deep learning model to detect <b>σ⁵⁴-dependent bacterial promoters</b> in DNA sequences.
    It scans your input using a sliding window of 81 base pairs, checking for promoter-like patterns.
    Matches are further validated against a known promoter database.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# === Text Input ===
user_input = st.text_area("🧬 Enter one or more DNA sequences (one per line):", height=200)

# === Predict Button ===
if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter at least one DNA sequence.")
    else:
        sequences = [seq.strip().upper() for seq in user_input.strip().split('\n') if seq.strip()]

        for idx, seq in enumerate(sequences, 1):
            if any(c not in "ATGC" for c in seq):
                st.error(f"❌ Sequence {idx} contains invalid characters. Skipping.")
                continue

            results = predict_sequence(seq)

            if results:
                st.markdown(f"<h4 style='color: #2e7d32;'>🔍 Results for Sequence {idx}</h4>", unsafe_allow_html=True)
                for sub_seq, prob, status in results:
                    if status == "Known Promoter":
                        bg_color = "#c8e6c9"
                        border = "#2e7d32"
                        icon = "✅"
                    else:
                        bg_color = "#e3f2fd"
                        border = "#1565c0"
                        icon = "🔎"

                    st.markdown(f"""
                        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 6px solid {border};">
                            {icon} <strong>{status}</strong><br>
                            <b>Confidence:</b> {prob:.2f}<br>
                            <b>81-mer:</b> <code>{sub_seq}</code>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="background-color: #ffebee; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 6px solid #e53935;">
                        ❌ <strong>No σ⁵⁴ Promoter detected</strong> in Sequence {idx}.
                    </div>
                """, unsafe_allow_html=True)
