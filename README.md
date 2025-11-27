EEG Avalanche Analysis and ATM Computation

This project processes EEG patient data to compute neuronal avalanche features and Avalanche Transition Matrices (ATM). It includes signal binarization, avalanche detection, feature extraction, and visualization.

Project Structure
project/
│
├── main.py             # Main script to load data, compute avalanches, ATM, and feature matrix
├── helpers.py          # Helper functions for data loading, preprocessing, avalanche detection, ATM computation   
└── atm_plots/          # Output folder for ATM heatmaps

Requirements

Python ≥ 3.8

NumPy

SciPy

Matplotlib

scipy.io (for .mat file reading)

Install dependencies with:

pip install numpy scipy matplotlib

Usage

Place all patient data in a folder (e.g., Z:\acutestroke_data_combineflipping_final\flipped_rightlesion).
Each patient folder should have the structure:

patient_id/
    patient_id_T1_RS_Eyes_Open_6_ICAclean/
        *.mat



Outputs

atm_plots/ contains ATM heatmaps for each patient.


Notes

Ensure that each patient has .mat files with Value arrays of shape (n_regions, n_samples).

bin_size depends on your sampling frequency (fs) and desired temporal resolution.

The code sorts patient IDs alphabetically to maintain consistent feature order.