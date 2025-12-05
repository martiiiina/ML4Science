# ML4Science: Avalanche and ATM Analysis

This repository contains code for preprocessing, feature extraction, and analysis of patient neural recordings. The focus is on detecting **neuronal avalanches** and computing **Avalanche Transition Matrices (ATMs)** from multi-region EEG time-series.

## Project Structure

```
ML4Science/
├── atm_plots/          # Heatmaps of ATM matrices for each patient
├── coh_plots/          # Heatmaps of Coh matrices for each patient
├── dataset_info/       # Information on data acquisition
├── extract_atm.py      # Extract ATMs and save features in a .csv file
├── extract_coh.py      # Extract Cohs and save features in a .csv file
├── helpers.py          # Helper functions for loading, binarization, ATM/Coh computation
├── knn_classifier.py/  # KNN classifier
├── Literature/         # Reference papers
├── README.md           # Project description and instructions
├── rf_classifier.py/   # Random Forest classifier
├── svm_classifier.py/  # SVM classifier
└── xgb_classifier.py/  # XGBoost classifier
```

## Requirements

Python 3.8+ and the following packages:

- numpy  
- scipy  
- matplotlib  
- scikit-learn

Install dependencies:

```
pip install numpy scipy matplotlib
```

## Data Loading

Patient data must follow this structure:

```
root/
├── Patient_001/
│   └── scout_data/
│       ├── epoch_001.npy
│       ├── epoch_002.npy
│       └── ...
├── Patient_002/
│   └── ...
```

Each `.npy` file contains **one EEG trial** with shape `(n_samples, n_channels)`.  
The loader automatically **transposes** arrays to `(channels, time)` and concatenates multiple trials **along the time axis**.

### load_all_patients()

Located in `helpers.py`, it:

- scans all patient directories  
- enters each `scout_data` folder  
- loads all `.npy` trials  
- sorts files to preserve trial order  
- concatenates trials along time  
- returns a dictionary:

```
{ "Patient_001": array(shape = regions × total_time), ... }
```

## Processing Pipeline

For each patient, `extract_atm.py` performs:

1. Z-score normalization across time  
2. Binarization using a Z-threshold  
3. Time binning (default: 4 ms → 20 samples at 5 kHz)  
4. Avalanche detection (active consecutive bins)  
5. Avalanche feature computation  
6. ATM construction (Avalanche Transition Matrix)  
7. Saving ATM heatmaps into `atm_plots/`

## Avalanche Features

`compute_avalanche_features()` returns:

- mean_size  
- max_size  
- mean_duration  
- max_duration  
- branching_factor  

Definitions:  
- **size** → total active region-bins  
- **duration** → number of consecutive active bins  
- **branching factor** → average ratio n(t+1)/n(t)

## Avalanche Transition Matrices (ATMs)

`compute_ATM()`:

- counts transitions i → j across avalanche bins  
- normalizes rows to probabilities  
- averages across avalanches  
- produces a **patient-specific ATM**  

ATM heatmaps are saved in `atm_plots/`.

## Feature Matrix for Machine Learning

`build_feature_matrix()` creates an array:

```
(n_patients × n_regions²)
```

Each row stores the **flattened ATM** of a patient.

## Usage

Run the full pipeline:

```
python extract_atm.py
```

This will:

- load `.npy` trials  
- perform normalization + binning + avalanche detection  
- compute patient ATMs  
- save ATM heatmaps  
- build the feature matrix `X`  

## Output

- `atm_plots/*.png` — ATM heatmaps  
- `X.npy` (optional) — feature matrix  
- console logs with ATM statistics and avalanche counts  

## Author

Project maintained by **Martina** for the EPFL ML4Science course.  
Feel free to open issues or request improvements!
