import os
import tempfile
import streamlit as st
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.integrate import trapezoid
from PIL import Image
import pandas as pd

# --- Streamlit Config ---
st.set_page_config(page_title="EEG Cognitive Load", layout="wide")

# --- Set Background ---
def set_bg(image_file):
    with open(image_file, "rb") as file:
        encoded = file.read()
    b64 = base64.b64encode(encoded).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .big-font {{
            font-size:35px !important;
            font-weight: bold;
            color: white;
            text-shadow: 1px 1px 2px black;
        }}
        .stTextInput>div>div>input {{
            background-color: rgba(255,255,255,0.9);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Set your background image here
import base64
if os.path.exists("bg.png"):
    set_bg("bg.png")

# --- Title ---
st.markdown('<p class="big-font">🧠 EEG-Based Cognitive Load Classifier</p>', unsafe_allow_html=True)
st.write("Upload one or more EEG `.edf` files to analyze Alpha, Beta, Theta bands and classify cognitive load.")

# --- EEG Bands ---
BANDS = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
EEG_CHANNELS = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..',
    'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.',
    'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.',
    'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..',
    'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..'
]

# --- Cognitive Load Formula ---
def assess_cognitive_load(alpha, beta, theta):
    denom = alpha + beta
    if denom == 0:
        return "Low Load"
    ratio = theta / denom
    if ratio > 1.0:
        return "High Load"
    elif ratio < 0.5:
        return "Low Load"
    return "Medium Load"

# --- Band Power Computation ---
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
            idx = np.logical_and(freqs >= BANDS[band][0], freqs <= BANDS[band][1])
            return trapezoid(psd[idx], freqs[idx]) * 1e12  # µV²

        alpha_power.append(band_power('alpha'))
        beta_power.append(band_power('beta'))
        theta_power.append(band_power('theta'))
    return np.mean(alpha_power), np.mean(beta_power), np.mean(theta_power)

# --- File Upload ---
uploaded_files = st.file_uploader("Upload .edf Files", type="edf", accept_multiple_files=True)

if uploaded_files:
    results = []
    st.markdown("### 📊 EEG File Analysis")
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
            sfreq = raw.info['sfreq']
            raw.pick_channels([ch for ch in EEG_CHANNELS if ch in raw.ch_names])

            if len(raw.ch_names) == 0:
                st.warning(f"⚠️ No EEG channels found in {file.name}. Skipping.")
                continue

            alpha, beta, theta = compute_band_power(raw, sfreq)
            load = assess_cognitive_load(alpha, beta, theta)
            results.append((file.name, alpha, beta, theta, load))

            st.success(f"✅ {file.name}: α = {alpha:.2f}, β = {beta:.2f}, θ = {theta:.2f} → **{load}**")

        except Exception as e:
            st.error(f"❌ Failed to process {file.name}: {e}")
        finally:
            os.remove(tmp_path)

    # --- Show Data Table ---
    df = pd.DataFrame(results, columns=["File", "Alpha", "Beta", "Theta", "Cognitive Load"])
    st.markdown("### 🧾 Summary Table")
    st.dataframe(df)

    # --- Plot ---
    st.markdown("### 📈 Band Power Comparison")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["File"], df["Alpha"], marker='o', label="Alpha")
    ax.plot(df["File"], df["Beta"], marker='s', label="Beta")
    ax.plot(df["File"], df["Theta"], marker='^', label="Theta")
    ax.set_xlabel("EDF File")
    ax.set_ylabel("Power (µV²)")
    ax.set_title("EEG Band Power by File")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # --- Download Button ---
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results as CSV", csv, file_name="cognitive_load_results.csv", mime='text/csv')
else:
    st.info("Upload `.edf` files above to begin analysis.")
