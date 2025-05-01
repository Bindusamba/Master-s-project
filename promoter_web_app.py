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

# === One-hot Encoding ===
base_map = {'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'N': [0, 0, 0, 0]}

def one_hot_encode(seq):
    seq = seq.upper()
    seq = ''.join([base if base in base_map else 'N' for base in seq])
    return [base_map[base] for base in seq]

# === Prediction Function ===
def predict_known_promoters(seq):
    results = []
    if len(seq) < 81:
        return results  # too short to evaluate
    for i in range(len(seq) - 80):
        window = seq[i:i+81]
        encoded = np.array(one_hot_encode(window)).reshape(1, 81, 4)
        prob = model.predict(encoded, verbose=0)[0][0]
        if prob > 0.5 and window in promoter_df["PromoterSequence"].values:
            results.append((window, prob))
    return results

# === UI HEADER ===
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧬 σ⁵⁴ Known Promoter Identifier</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='background-color: #f0f9ff; border-left: 6px solid #2196f3; padding: 15px;
            border-radius: 10px; margin-bottom: 25px; font-family: "Helvetica", sans-serif;'>
<h4 style='color:#0b5394;'>📘 About This Tool</h4>
<p style='color:#333; font-size:16px; line-height:1.6;'>
This tool identifies <b>known σ⁵⁴ bacterial promoters</b> using a trained deep learning model. It uses a sliding window of 81 base pairs and only reports matches found in the curated promoter database.
</p>
</div>
""", unsafe_allow_html=True)

# === Text Input ===
user_input = st.text_area("🧬 Paste DNA sequences (one per line):", height=200)

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

            st.markdown(f"<h4 style='color: #333;'>🔬 Results for Sequence {idx}</h4>", unsafe_allow_html=True)

            if len(seq) < 81:
                st.markdown(f"""
                    <div style="background-color: #fff3cd; padding: 15px; border-radius: 10px;
                                border-left: 6px solid #ff9800; margin-bottom: 10px;">
                        ⚠️ Sequence too short for σ⁵⁴ promoter scanning (needs ≥81 bp).
                    </div>
                """, unsafe_allow_html=True)
                continue

            results = predict_known_promoters(seq)

            if results:
                for window, prob in results:
                    st.markdown(f"""
                        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px;
                                    margin-bottom: 10px; border-left: 6px solid #2e7d32;">
                            ✅ <strong>Known σ⁵⁴ Promoter Found</strong><br>
                            <b>Confidence:</b> {prob:.2f}<br>
                            <b>81-mer:</b> <code>{window}</code>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="background-color: #ffebee; padding: 15px; border-radius: 10px;
                                margin-bottom: 10px; border-left: 6px solid #e53935;">
                        ❌ <strong>No σ⁵⁴ Promoter identified</strong> in this sequence.
                    </div>
                """, unsafe_allow_html=True)
