# EEG Cognitive Load Classifier

## Overview

Cognitive load assessment is vital for understanding human attention, effort, and fatigue during tasks. This project classifies mental workload (**High**, **Medium**, **Low**) using non-invasive EEG signals, offering an objective alternative to traditional subjective surveys. EEG band powers (**Alpha**, **Beta**, **Theta**) are extracted and used in simple machine learning models for classification.

## Objectives

- Classify mental workload levels (High, Medium, Low) using EEG analysis (Alpha, Beta, Theta bands).
- Automate EEG data extraction, spectral analysis, and classification using Python.
- Visualize cognitive load trends across subjects and sessions.

## Methodology

### Pipeline Steps

#### Signal Acquisition
- Uses publicly available EEG datasets (e.g., Physionet).
- No hardware required; raw EEG files are input.

#### Preprocessing
- Band-pass filtering removes noise and isolates Alpha, Beta, Theta frequency bands.
- Uses the MNE Python library for EEG data loading and preprocessing.

#### Feature Extraction
- Computes Power Spectral Density (PSD) using Welch’s method.
- Integrates PSD for each frequency band (Alpha, Beta, Theta) using SciPy.

#### Classification
- Extracted band powers classified into High, Medium, or Low cognitive load.
- Classification is based on band power thresholds.

#### Visualization
- Plots EEG band powers for trend analysis across subjects using Matplotlib.

## Tools Used

- **Python**: Programming
- **MNE**: EEG data loading and preprocessing
- **SciPy**: Signal processing
- **Matplotlib**: Visualization
- **Welch’s Method**: Band power estimation

## Input & Output

| Aspect           | Description                                     |
|------------------|------------------------------------------------|
| Input            | Raw EEG files (Physionet dataset or similar)    |
| Processing       | Channel extraction, PSD computation, Alpha/Beta/Theta band integration |
| Terminal Output  | Band power values and workload classification per file |
| Plot Output      | Visual trend comparison (band power, across subjects) |

## Installation

Clone this repository:
git clone https://github.com/ramkumar27072006/EEG-Cognitive-Load-Classifier.git
cd EEG-Cognitive-Load-Classifier

text

Install required packages:
pip install -r requirements.txt

text
*(You may need to create or update `requirements.txt` with: `mne`, `scipy`, `matplotlib`, `numpy`.)*

## Usage

**Prepare EEG Data**
- Download raw EEG dataset (e.g., from Physionet).
- Place files in the designated `/data` directory.

**Run the Classifier**
python main.py --input data/<eeg_file>

text
- Outputs terminal classification (workload: High, Medium, Low) per file.
- Generates band power plots for visual analysis.

**Configuration**
- Script parameters (e.g., frequency bands, classification thresholds) can be adjusted in the configuration section of the code.

## Results

- **Terminal output**: Shows Alpha, Beta, Theta values for each file and classified cognitive load.
- **Graphs**: Band power across subjects for trend analysis.

## Applications

- Adaptive learning systems
- Mental workload monitoring in critical environments
- Stress detection in workplaces
- Mindfulness/meditation state tracking

## Credits

**Team:** Pragalya M, Ramkumar R, Youvashree K  
Based on Physionet EEG datasets.

## License

MIT License
