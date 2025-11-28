# ML4Science: Avalanche and ATM Analysis

This repository contains code for preprocessing, feature extraction, and analysis of patient neural recordings.  
The focus is on detecting **neuronal avalanches** and computing **Avalanche Transition Matrices (ATMs)** from multi-region EEG time-series.

---

## **Project Structure**

```
ML4Science/
├── atm_plots/          # Heatmaps of ATM matrices for each patient
├── dataset_info/       # Information on data acquisition
├── helpers.py          # Helper functions for loading, binarization, ATM computation
├── main.py             # Main script for processing all patients
├── Literature/         # Reference papers
└── README.md           # Project description and instructions
```

---

## **Requirements**

Python 3.8+ and the following packages:

- `numpy`  
- `scipy`  
- `matplotlib`  
- `h5py` / `scipy.io` (for reading .mat files)

Install dependencies:

```bash
pip install numpy scipy matplotlib h5py
```

---

## **Data Loading**

Patient data must follow this structure:

```
root/
├── Patient_001/
│   └── Patient_001_T1_RS_Eyes_Open_6_ICAclean/
│       ├── epoch1.mat
│       ├── epoch2.mat
│       └── ...
├── Patient_002/
│   └── ...
```

The function `load_all_patients()` in `helpers.py`:

- scans all patient folders  
- loads `.mat` epochs  
- concatenates them along time  
- returns:  
  ```
  { "Patient_001": np.array(regions × time), ... }
  ```

---

## **Processing Pipeline**

For each patient, `main.py` performs:

1. **Z-score normalization** along time for each region  
2. **Binarization** using a Z-threshold  
3. **Time binning** (default bin = 4 ms → 20 samples at 5 kHz)  
4. **Avalanche detection**: sequences of consecutive active bins  
5. **Computing avalanche features**  
6. **Building the ATM** (Avalanche Transition Matrix)  
7. **Saving ATM heatmaps**

---

## **Avalanche Features**

The function `compute_avalanche_features()` computes:

- `mean_size`
- `max_size`
- `mean_duration`
- `max_duration`
- `branching_factor`

Where:

- **size** = total number of active region-bins  
- **duration** = number of consecutive bins  
- **branching factor** = average of *n(t+1) / n(t)* across bins  

---

## **ATMs (Avalanche Transition Matrices)**

`compute_ATM()` computes transition probabilities:

- for each avalanche, counts transitions *i → j* from bin t to t+1  
- normalizes rows into a probability matrix  
- averages across avalanches → **patient ATM**

ATM plots are saved automatically into `atm_plots/`.

---

## **Feature Matrix for Machine Learning**

`build_feature_matrix()` converts all patient ATMs into a single `(n_patients × n_regions²)` array:

```
X[i] = flatten(ATM_of_patient_i)
patient_ids[i] = patient label
```

This prepares features for ML classifiers or clustering.

---

## **Usage**

Run the full pipeline:

```bash
python main.py
```

This will:

- load all patients  
- run normalization + binning + avalanche detection  
- compute ATM for each patient  
- save ATM heatmaps  
- build feature matrix `X`  

---

## **Output**

- `atm_plots/*.png` — heatmaps per patient  
- `X.npy` (if saved by user) — feature matrix  
- console logs with avalanche counts and ATM details  

---

## **Author**

Project maintained by **Martina** for the EPFL ML4Science course.

Feel free to open issues or ask for improvements!
