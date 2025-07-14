import streamlit as st
import os
import numpy as np
import mne
from scipy.signal import welch
from scipy.integrate import trapezoid
import pandas as pd
import matplotlib.pyplot as plt

# --- Background & CSS ---
def set_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('bg.png');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
        }}
        .title-font {{
            font-size: 40px;
            font-weight: bold;
            text-shadow: 2px 2px 5px #000;
        }}
        .metric-box {{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 1rem;
            border-radius: 10px;
            margin: 10px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- EEG Band Info ---
BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}

EEG_CHANNELS = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..',
    'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.',
    'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.',
    'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..',
    'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..'
]

def compute_band_power(raw, sfreq):
    alpha_power = []
    beta_power = []
    theta_power = []

    for ch in EEG_CHANNELS:
        if ch not in raw.ch_names:
            continue
        data, _ = raw.copy().pick(ch).get_data(return_times=True)
        freqs, psd = welch(data[0], sfreq, nperseg=min(1024, len(data[0])))

        def band_power(band):
            low, high = BANDS[band]
            idx = np.logical_and(freqs >= low, freqs <= high)
            return trapezoid(psd[idx], freqs[idx]) * 1e12

        alpha_power.append(band_power('alpha'))
        beta_power.append(band_power('beta'))
        theta_power.append(band_power('theta'))

    return np.mean(alpha_power), np.mean(beta_power), np.mean(theta_power)

def assess_cognitive_load(alpha, beta, theta):
    if (alpha + beta) == 0:
        return "Low"
    ratio = theta / (alpha + beta)
    if ratio > 1.0:
        return "High"
    elif ratio < 0.5:
        return "Low"
    return "Medium"

# --- UI Starts ---
set_bg()
st.markdown('<div class="title-font">🧠 EEG-Based Cognitive Load Classifier</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload EEG EDF file(s)", type=["edf"], accept_multiple_files=True)

results = []

if uploaded_files:
    for file in uploaded_files:
        with st.spinner(f"Processing {file.name}..."):
            raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
            sfreq = raw.info['sfreq']
            raw.pick_channels([ch for ch in EEG_CHANNELS if ch in raw.ch_names])

            alpha, beta, theta = compute_band_power(raw, sfreq)
            load = assess_cognitive_load(alpha, beta, theta)

            results.append({
                "File": file.name,
                "Alpha": round(alpha, 2),
                "Beta": round(beta, 2),
                "Theta": round(theta, 2),
                "Load": load
            })

            st.markdown(f"<div class='metric-box'><b>{file.name}</b><br>α: {alpha:.2f} | β: {beta:.2f} | θ: {theta:.2f} → <b>{load} Load</b></div>", unsafe_allow_html=True)

    # Show plot button
    if st.button("📈 Show EEG Band Power Plot"):
        st.subheader("Band Power Comparison")
        df = pd.DataFrame(results)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['File'], df['Alpha'], marker='o', label='Alpha')
        ax.plot(df['File'], df['Beta'], marker='s', label='Beta')
        ax.plot(df['File'], df['Theta'], marker='^', label='Theta')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Power (μV²)")
        plt.title("EEG Band Power")
        plt.legend()
        st.pyplot(fig)

    # Save to CSV
    if st.button("💾 Export Results to CSV"):
        df = pd.DataFrame(results)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, file_name="eeg_cognitive_load.csv", mime="text/csv")
