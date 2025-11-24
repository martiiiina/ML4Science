Acute Stroke EEG Data (Converted to .npy Format)
Overview

This dataset contains EEG recordings from acute stroke patients.
Originally provided as MATLAB .mat files, all data have been pre-converted into NumPy .npy files to speed up loading and analysis in Python.

Each patient has multiple EEG epochs saved as:

One .npy file per epoch

Shape: (62 regions × 30000 timepoints)

Stored inside a folder: npy_data/

Folder Structure
root/
│
├── TiMeS_WP11_001/
│   └── TiMeS_WP11_001_T1_RS_Eyes_Open_6_ICAclean/
│       ├── npy_data/
│       │   ├── TiMeS_WP11_001_T1_RS_Eyes_Open_6_ICAclean_1_DKT_mean.npy
│       │   ├── TiMeS_WP11_001_T1_RS_Eyes_Open_6_ICAclean_2_DKT_mean.npy
│       │   └── ...
│
├── TiMeS_WP11_002/
│   └── TiMeS_WP11_002_T1_RS_Eyes_Open_6_ICAclean/
│       ├── npy_data/
│       │   ├── TiMeS_WP11_002_T1_RS_Eyes_Open_6_ICAclean_1_DKT_mean.npy
│       │   └── ...
│
└── ...

Key Points

Patient folder: TiMeS_WP11_XXX

Internal EEG folder: {TiMeS_WP11_XXX}_T1_RS_Eyes_Open_6_ICAclean

Converted .npy epoch files live inside npy_data/

Each epoch corresponds to one .npy file

.npy File Structure

Each .npy file contains:

Shape: (62, 30000)

Rows: 62 cortical regions (DK atlas)

Columns: 30000 timepoints sampled at 5000 Hz (6 seconds total)

Stored as a plain NumPy float array for ultra-fast loading.

Python Loading Scripts
load_patient(npy_folder)

Loads all epochs for a single patient.

Input: path to the patient's npy_data/ folder

Output: list of NumPy arrays

Each array is: (62 × 30000)

Example Usage
root = r"Z:\acutestroke_data_combineflipping_final\flipped_rightlesion"

from helpers import load_all_patients

patients = load_all_patients(root)
