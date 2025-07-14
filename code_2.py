import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.integrate import trapezoid

# Parent EEG folder
EEG_FOLDER = r"D:\3rd\proj\BCI_dataset"

# Frequency bands (Hz)
BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}

# EEG channels of interest
EEG_CHANNELS = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..',
    'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.',
    'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.',
    'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..',
    'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..'
]

# Cognitive load classification (θ / (α + β) method)
def assess_cognitive_load(alpha, beta, theta):
    denominator = alpha + beta
    if denominator == 0:
        return "Low Load"
    theta_ratio = theta / denominator
    if theta_ratio > 1.0:
        return "High Load"
    elif theta_ratio < 0.5:
        return "Low Load"
    else:
        return "Medium Load"

# Band power computation
def compute_band_power(raw, sfreq):
    alpha_power = []
    beta_power = []
    theta_power = []

    for ch in EEG_CHANNELS:
        if ch not in raw.ch_names:
            continue
        data, _ = raw.copy().pick(ch).get_data(return_times=True)
        freqs, psd = welch(data[0], sfreq, nperseg=1024)

        def band_power(band):
            low, high = BANDS[band]
            idx = np.logical_and(freqs >= low, freqs <= high)
            return trapezoid(psd[idx], freqs[idx]) * 1e12  # µV²

        alpha_power.append(band_power('alpha'))
        beta_power.append(band_power('beta'))
        theta_power.append(band_power('theta'))

    return np.mean(alpha_power), np.mean(beta_power), np.mean(theta_power)

# Collect all .edf files from all subfolders
results = []
for root, dirs, files in os.walk(EEG_FOLDER):
    edf_files = [f for f in files if f.endswith('.edf')]
    subject = os.path.basename(root)

    for file in sorted(edf_files):
        path = os.path.join(root, file)
        print(f"\n📂 Processing: {subject}/{file}")

        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        raw.pick_channels([ch for ch in EEG_CHANNELS if ch in raw.ch_names])

        if len(raw.ch_names) == 0:
            print("⚠️ No valid EEG channels. Skipping.")
            continue

        print(f"🧠 Channels used: {len(raw.ch_names)}")
        alpha, beta, theta = compute_band_power(raw, sfreq)
        load = assess_cognitive_load(alpha, beta, theta)

        results.append((subject, file, alpha, beta, theta, load))
        print(f"{file:20} α: {alpha:8.2f} β: {beta:8.2f} θ: {theta:8.2f} => {load}")

# Print summary
print("\n--- Cognitive Load Summary ---")
print(f"{'Subject':8} {'File':15} {'Alpha':>10} {'Beta':>10} {'Theta':>10} {'Load'}")
for r in results:
    print(f"{r[0]:8} {r[1]:15} {r[2]:10.2f} {r[3]:10.2f} {r[4]:10.2f} {r[5]}")

# ✅ Plot combined band powers for all subjects
import matplotlib.pyplot as plt

subjects = [f"{r[0]}/{r[1]}" for r in results]
alpha_vals = [r[2] for r in results]
beta_vals = [r[3] for r in results]
theta_vals = [r[4] for r in results]

plt.figure(figsize=(14, 6))
plt.plot(subjects, alpha_vals, marker='o', label='Alpha')
plt.plot(subjects, beta_vals, marker='s', label='Beta')
plt.plot(subjects, theta_vals, marker='^', label='Theta')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Subject/File')
plt.ylabel('Power (µV²)')
plt.title('EEG Band Power Across Subjects')
plt.legend()
plt.tight_layout()
plt.show()
