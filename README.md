# ML4Science: Avalanche and ATM Analysis

This repository contains code for preprocessing, feature extraction, and analysis of patient neural recordings. The focus is on detecting **neuronal avalanches** and computing **Avalanche Transition Matrices (ATMs)** from multi-region EEG time-series.

## Project Structure
ML4Science/
├── atm_plots/          # Heatmaps of ATM matrices for each patient
├── dataset_info/       # Information on data acquisition
├── helpers.py          # Helper functions for loading, binarization, ATM computation
├── main.py             # Main script for processing all patients
├── Literature/         # Reference papers
└── README.md           # Project description and instructions

## Requirements
Python 3.8+ and the following packages:
- numpy
- scipy
- matplotlib

Install:
pip install numpy scipy matplotlib

## Data Loading
Patient data must follow this structure:
root/
├── Patient_001/
│   └── scout_data/
│       ├── epoch_001.npy
│       ├── epoch_002.npy
│       └── ...
├── Patient_002/
│   └── ...

Each `.npy` file contains **one EEG trial** with shape `(n_samples, n_channels)`. The loader automatically transposes arrays to `(channels, time)` and concatenates all trials along the time axis.

### load_all_patients()
Located in `helpers.py`, it:
- scans all patient folders
- enters the `scout_data` subfolder
- loads all `.npy` trials
- sorts files to preserve trial order
- concatenates trials along time
- returns a dict:
  { "Patient_001": array(regions × total_time), ... }

## Processing Pipeline
For each patient, `main.py` performs:
1. Z-score normalization across time for each region  
2. Binarization using a Z-threshold  
3. Time-binning (default: 4 ms → 20 samples at 5 kHz)  
4. Avalanche detection (consecutive active bins)  
5. Avalanche feature computation  
6. ATM construction (Avalanche Transition Matrix)  
7. Saving ATM heatmaps in `atm_plots/`

## Avalanche Features
`compute_avalanche_features()` returns:
- mean_size
- max_size
- mean_duration
- max_duration
- branching_factor  
(size = total active region-bins, duration = number of consecutive bins, branching factor = avg n(t+1)/n(t))

## Avalanche Transition Matrices (ATMs)
`compute_ATM()`:
- counts transitions i→j within avalanches
- normalizes rows into a probability matrix
- averages across avalanches
The resulting ATM is saved as a heatmap.

## Feature Matrix for Machine Learning
`build_feature_matrix()` creates an array:
(n_patients × n_regions²)
Each row = flattened ATM of a patient.

## Usage
Run:
python main.py
This will:
- load all `.npy` trials
- run normalization + binning + avalanche detection
- compute ATM per patient
- save heatmaps
- build the feature matrix X

## Output
- atm_plots/*.png — ATM heatmaps  
- X.npy (optional) — feature matrix  
- console logs with avalanche and ATM statistics

## Author
Project maintained by **Martina** for the EPFL ML4Science course. Feel free to open issues or request improvements!
