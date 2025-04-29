import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your trained CNN model
model = load_model("cnn_model.h5")

# One-hot encoding map
base_map = {'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'N': [0, 0, 0, 0]}

def one_hot_encode(sequence, max_len=81):
    sequence = sequence.upper()
    sequence = ''.join([base if base in base_map else 'N' for base in sequence])
    if len(sequence) < max_len:
        sequence = sequence + 'N' * (max_len - len(sequence))  # pad
    else:
        sequence = sequence[:max_len]  # truncate
    encoded = [base_map[base] for base in sequence]
    return np.array(encoded)

def predict_promoter(seq):
    encoded = one_hot_encode(seq)
    X = np.expand_dims(encoded, axis=0)
    y_pred = model.predict(X)[0][0]
    return y_pred

def highlight_promoter(sequence):
    # Highlight TTGGCACG as sample sigma54 motif (example; adjust to your model's learned features)
    motif = "TTGGCACG"
    seq = sequence.upper()
    if motif in seq:
        start = seq.find(motif)
        end = start + len(motif)
        highlighted = seq[:start] + f"<span style='background-color: #FFD700; color: black'>{seq[start:end]}</span>" + seq[end:]
        return highlighted
    else:
        return seq

# Streamlit UI
st.title("σ54 Promoter Detection Web App")
st.write("Paste one or more DNA sequences (one per line):")

user_input = st.text_area("DNA Sequences", height=300, placeholder="ATGCTACGT...")

if st.button("Predict σ54 Promoter"):
    if not user_input.strip():
        st.warning("Please enter at least one DNA sequence.")
    else:
        sequences = user_input.strip().split("\n")
        results = []
        for i, seq in enumerate(sequences):
            if not seq.strip():
                continue
            pred = predict_promoter(seq)
            label = "Promoter" if pred > 0.5 else "Non-Promoter"
            color_seq = highlight_promoter(seq)
            results.append({
                "Sequence #": i + 1,
                "Prediction": label,
                "Probability": round(pred, 4),
                "Highlighted Sequence": color_seq
            })

        df = pd.DataFrame(results)
        for idx, row in df.iterrows():
            st.markdown(f"### Sequence {row['Sequence #']}")
            st.markdown(f"**Prediction**: {row['Prediction']} ({row['Probability']})")
            st.markdown("**Highlighted:**", unsafe_allow_html=True)
            st.markdown(f"<code>{row['Highlighted Sequence']}</code>", unsafe_allow_html=True)
