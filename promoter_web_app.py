import streamlit as st
import numpy as np
import pandas as pd
import re
from tensorflow.keras.models import load_model

# === PAGE CONFIG ===
st.set_page_config(page_title="σ⁵⁴ Promoter Predictor", page_icon="🧬", layout="centered")

# === Load Model & Promoter Database ===
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

# === Fuzzy σ⁵⁴ Motif Checker ===
def is_sigma54_motif(seq):
    """
    Check for σ⁵⁴ promoter-like patterns allowing small deviations.
    Looks for TGGCxx near -24 and TTGCx near -12 with ~7 bp spacing.
    """
    for i in range(10, 20):  # sliding between positions
        part1 = seq[i:i+6]         # potential -24 box
        spacer = seq[i+6:i+13]     # 7 bp spacer
        part2 = seq[i+13:i+18]     # potential -12 box

        if len(part1) < 6 or len(part2) < 5:
            continue

        mismatch1 = sum([a != b for a, b in zip(part1, 'TGGCAC')])
        mismatch2 = sum([a != b for a, b in zip(part2[:5], 'TTGCA')])

        if mismatch1 <= 1 and mismatch2 <= 1:
            return True
    return False

# === Predict Promoters in Any Length Sequence ===
def predict_sequence(seq, threshold=0.8):
    results = []
    for i in range(len(seq) - 80):  # sliding window of 81 bp
        window = seq[i:i+81]
        encoded = np.array(one_hot_encode(window)).reshape(1, 81, 4)
        prob = model.predict(encoded, verbose=0)[0][0]

        if prob > threshold and is_sigma54_motif(window):
            is_known = window in promoter_df["PromoterSequence"].values
            function = "Known promoter" if is_known else "Unknown promoter"
            results.append((window, prob, function, i))
    return results

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
    This web app detects <b>σ⁵⁴-dependent bacterial promoters</b> in any DNA sequence. It uses a deep learning model trained on 81 bp windows and filters results using the known motif <code>TGGCAC-N₇-TTGCW</code> (with tolerance for minor mismatches).
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# === Text Input ===
user_input = st.text_area("🧬 Enter DNA sequences (one per line, any length):", height=200)

# === Predict Button ===
if st.button("🔍 Predict"):
    sequences = [seq.strip().upper() for seq in user_input.strip().split('\n') if seq.strip()]
    
    for idx, seq in enumerate(sequences, 1):
        if any(c not in "ATGC" for c in seq):
            st.error(f"❌ Sequence {idx} contains invalid characters. Skipping.")
            continue
        elif len(seq) < 81:
            st.warning(f"⚠️ Sequence {idx} is shorter than 81 bp. Skipping.")
            continue

        results = predict_sequence(seq, threshold=0.8)

        if results:
            st.markdown(f"<h4 style='color: #2e7d32;'>🔍 Results for Sequence {idx}</h4>", unsafe_allow_html=True)
            for sub_seq, prob, function, start_idx in results:
                end_idx = start_idx + 81
                st.markdown(f"""
                    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 6px solid #66bb6a;">
                        ✅ <strong>σ⁵⁴ Promoter found</strong><br>
                        <b>Confidence:</b> {prob:.2f}<br>
                        <b>Position:</b> {start_idx}–{end_idx}<br>
                        <b>Matched 81-mer:</b> <code>{sub_seq}</code><br>
                        <b>Database Match:</b> {function}<br>
                        <b>Sequence with Promoter Highlighted:</b><br>
                        <div style="font-family: monospace; word-wrap: break-word; background-color: #f9fbe7; padding: 10px; border-radius: 5px;">
                        {seq[:start_idx]}<mark style="background-color: #a5d6a7; font-weight: bold;">{seq[start_idx:end_idx]}</mark>{seq[end_idx:]}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color: #fff8e1; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 6px solid #fbc02d;">
                    ⚠️ <strong>No σ⁵⁴ Promoter detected</strong> in Sequence {idx}.
                </div>
            """, unsafe_allow_html=True)