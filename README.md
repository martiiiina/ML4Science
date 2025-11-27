# ML4Science: Avalanche and ATM Analysis

This repository contains code for preprocessing, feature extraction, and analysis of patient neural recordings. The focus is on computing **avalanches** and **Avalanche Transition Matrices (ATM)** from multi-region time-series data.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Data Loading](#data-loading)
- [Processing Pipeline](#processing-pipeline)
- [Features and ATM](#features-and-atm)
- [Usage](#usage)
- [Output](#output)
- [Author](#author)

---

## Project Structure

ML4Science/
├── atm_plots/ # Heatmaps of ATM matrices for each patient
├── dataset_info/ # Raw data and metadata
├── helpers.py # Helper functions for loading, binarization, ATM computation
├── main.py # Main script for processing patients
├── Literature/ # Reference papers
├── README.md # Project description and instructions

## Requirements

Python 3.8+ with the following packages:

- numpy
- scipy
- matplotlib
- h5py / scipy.io (for `.mat` files)

Install via pip:

```bash
pip install numpy scipy matplotlib h5py
Data Loading
Patient data should be organized as follows:

root/
├── Patient_001/
│   └── Patient_001_T1_RS_Eyes_Open_6_ICAclean/
│       ├── epoch1.mat
│       ├── epoch2.mat
│       └── ...
├── Patient_002/
│   └── ...
The load_all_patients() function in helpers.py automatically reads all .mat files, concatenates epochs, and returns a dictionary {patient_id: np.array}.

Processing Pipeline
Z-score normalization along time for each region.

Binarization based on a Z-threshold.

Time binning: divide signal into bins of configurable size (default 4 ms).

Avalanche detection: sequences of contiguous active bins.

Feature computation: mean/max size, mean/max duration, branching factor.

ATM computation: transition probability matrices across regions.

Features and ATM
compute_avalanche_features(): returns a dictionary with:

mean_size

max_size

mean_duration

max_duration

branching_factor

compute_ATM(): returns the average Avalanche Transition Matrix (ATM) for a patient.

build_feature_matrix(): converts all patient ATMs into a matrix suitable for machine learning.

