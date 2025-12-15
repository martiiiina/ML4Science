# ML4Science: Avalanche and ATM Analysis

This repository contains code for preprocessing, feature extraction, and analysis of patient neural recordings. The focus is on detecting **neuronal avalanches** and computing **Avalanche Transition Matrices (ATMs)** from multi-region EEG time-series.

## Project Structure

```
ML4Science/
├── Literature/                              # Reference papers
├── atm_plots/                               # Heatmaps of ATM matrices for each patient
├── coh_plots/                               # Heatmaps of Coh matrices for each patient
├── dataset_info/                            # Information on data acquisition
├── shap_plots/                              # Shap values plots
├── shap_values/                             # Shape values stored in .csv files for all seeds
├── xgb__best_models/                        # Best XGB model for each random state, used for SHAP analysis
├── README.md                                # Project description and instructions
├── atm_dataset.csv                          # DataFrame of ATM features
├── coh_dataset.csv                          # DataFrame of Coh features
├── extract_atm.py                           # Extract ATMs and save features in a .csv file
├── extract_coh.py                           # Extract Cohs and save features in a .csv file
├── helpers.py                               # Helper functions for loading, binarization, ATM/Coh computation
├── optimize_bf.py                           # Script for time binning optimization
├── results_atm.xlsx                         # Bbalanced accuracies from ATM classification
├── results_coh.xlsx                         # Balanced accuracies from Coh classification
├── rf_classifier.py                         # Random Forest classifier
├── shap_analysis.py                         # Script to perform SHAP analysis on XGB
├── shap_values_atm_regions_all_seeds.xlsx   # Summary of SHAP analysis
├── statistical_analysis.jpynb               # Statistial analysis on NA scalar features
├── svm_classifier.py                        # SVM classifier
└── xgb_classifier.py                        # XGBoost classifier
```

## Requirements

Python 3.8+ and the following packages:

- numpy  
- pandas
- scipy  
- matplotlib  
- seaborn
- scikit-learn
- statsmodels
- mne
- mne-connectivity
- imblearn
- xgboost
- shap
- joblib
- pathlib

Install dependencies:

```
pip install numpy pandas scipy matplotlib seaborn statsmodels scikit-learn mne mne-connectivity imbalanced-learn xgboost shap joblib pathlib
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
# Neuronal Avalanches, ATM, and Coherence Analysis Pipeline

This pipeline processes patient EEG data to extract **neuronal avalanches**, compute **Avalanche Transition Matrices (ATMs)**, and calculate **functional coherence** features. The output is a tabular dataset suitable for machine learning or statistical analysis. 
The data used in this project belongs to the UPHummel Lab. To run the scripts, you can start with the classification part by using the pre-processed tabular datasets, which already contain the extracted features. These datasets are ready for analysis and machine learning tasks, so there's no need to repeat the preprocessing steps. Hereafter, all the steps followed for the analysis are anyway illustrated. 

---

## Overview

The pipeline consists of two main parts:

1. **Avalanche and ATM Extraction**  
   - Load multi-region EEG data from `.npy` files for each patient.
   - Perform **Z-score normalization** across time for each brain region.
   - **Binarize** signals according to a z-threshold to detect active events.
   - **Time-binning**: group time points into bins (default: 4 ms).
   - **Avalanche detection**: identify consecutive active bins across regions.
   - Compute **avalanche features**: mean size, max size, mean duration, max duration, branching factor.
   - Construct **Avalanche Transition Matrices (ATMs)**: probability of transitions between active regions in consecutive bins.
   - Save ATM heatmaps in `atm_plots/`.
   - Flatten ATMs into feature vectors and build a **patient-level dataframe** (`atm_dataset.csv`).

2. **Coherence Feature Extraction**  
   - Load patient EEG data as epochs without concatenation.
   - Create **MNE EpochsArray** objects for each patient.
   - Compute **spectral coherence** in a frequency band (default: 8–12 Hz) using `mne_connectivity.spectral_connectivity_time`.
   - Extract the **full coherence matrix**, symmetrize it, and save heatmaps in `coh_plots/`.
   - Flatten the **lower triangular part** of the coherence matrix to create feature vectors.
   - Apply **Z-score normalization** to coherence features.
   - Build a **patient-level dataframe** (`coherence_dataset.csv`).

3. **Branching factor optimization**
   - Coomputes branching factors for different candidate bin sizes
   - Selects the bin size that minimizes the norm of the difference between observed branching factors and 1 (i.e. criticality condition)

## Usage

Run the full pipeline:

```
python extract_atm.py        # For ATM and avalanche feature extraction
python extract_coh.py        # For coherence feature extraction
```

## Machine Learning Classification

The repository provides scripts to train and evaluate machine learning classifiers on the extracted features:

- `svm_classifier.py` — Support Vector Machine (SVM)
- `rf_classifier.py`  — Random Forest (RF)
- `xgb_classifier.py` — XGBoost (XGB)

### Pipeline Overview

Each script follows the same general pipeline:

1. **Load feature dataset**  
   - For ATM features: `atm_dataset.csv`  
   - For coherence features: `coherence_dataset.csv` 

2. **Split data**  
   - Stratified train-test split (default 80/20)  
   - Maintains class distribution  

3. **Preprocessing and balancing**  
   - Random undersampling of majority class  
   - SMOTE oversampling of minority class  
   - Feature scaling with `StandardScaler`  

4. **Model training**  
   - Grid search with cross-validation (`StratifiedKFold`) to find best hyperparameters  
   - Scoring metric: **balanced accuracy**  

5. **Evaluation**  
   - Test set balanced accuracy  
   - Full classification report  

6. **Results saving**  
   - Best hyperparameters, CV and test balanced accuracy are saved to `results.xlsx` or `results_coherence.xlsx`  
   - For multiple random seeds, a final summary with mean and standard deviation of test balanced accuracy is printed  

### Example Usage

To classify stroke vs. healthy EEG, you can select the feature of interest (either **Avalanche Transition Matrices (ATM)** or **Coherence**) by modifying the following lines in the classifier scripts:

- Change the dataset loaded from **atm_dataset.csv** to **coherence_dataset.csv** depending on the feature you want to use:
  ```python
  df = pd.read_csv("atm_dataset.csv")  # For ATM features
  df = pd.read_csv("coherence_dataset.csv")  # For Coherence features

- Change the feature selection line to match the feature type:
  ```python
  X = df.filter(regex="^atm_").values  # For ATM features
  X = df.filter(regex="^coh_").values  # For Coherence features

- Similarly, change the result filename when saving the classification output:
  ```python
   save_results_to_excel(
      model_name="SVM",
      best_params=grid.best_params_,
      cv_score=grid.best_score_,
      test_score=test_bal_acc,
      filename="results_atm.xlsx"  # For ATM features
   )
   save_results_to_excel(
      model_name="SVM",
      best_params=grid.best_params_,
      cv_score=grid.best_score_,
      test_score=test_bal_acc,
      filename="results_coherence.xlsx"  # For Coherence features
   )

Run the SVM classifier over 50 random seeds:

```
python svm_classifier.py
```

## SHAP Analysis Across Multiple Random Seeds

This repository performs a comprehensive **SHAP (SHapley Additive exPlanations) analysis** to evaluate the **stability and distribution of feature importance across 50 random seeds** for XGBoost models trained on ATM-based features. The workflow quantifies how consistent model explanations are across different random initializations and maps feature-level SHAP values back to **ATM brain regions (ROIs)** for interpretability.

### Overview of the Analysis

The pipeline addresses three key questions:

1. **Are SHAP feature importances stable across random seeds?**
2. **Which ATM features and ROI-to-ROI connections are consistently influential?**
3. **Which brain regions (nodes) show the highest overall importance across all seeds?**

To answer these, the script:
- Computes SHAP values for each trained model (one per seed)
- Aggregates SHAP statistics across seeds
- Visualizes distributions using bar plots, beeswarm plots, and heatmaps
- Reconstructs SHAP values into ROI-to-ROI matrices
- Exports detailed results to CSV and Excel for downstream analysis

### Directory Structure

The script automatically creates the following output folders:

```text
shap_plots/
│── shap_distribution_across_seeds.png
│── shap_beeswarm_seed_*.png
│── shap_beeswarm_aggregated_all_seeds.png
│── shap_bar_aggregated_all_seeds.png
│── shap_atm_heatmap_aggregated.png

shap_values/
│── shap_values_regions_seed*.csv
│── shap_values_atm_regions_all_seeds.csv
│── shap_values_atm_regions_all_seeds.xlsx
```

### Data and Model Inputs

### Dataset
- **File**: `atm_dataset.csv`
- **Features**: All columns whose names start with `atm_`
- **Target variable**: `label`
- **Feature dimensionality**:
  - Each subject is represented by a flattened ATM connectivity matrix
  - Features correspond to pairwise connections between **62 brain regions**

### Models
- **Directory**: `xgb_models/`
- **Filename pattern**:
  ```text
  best_xgb_model_seed_{seed}.pkl


## Author

Project maintained by **Martina** for the EPFL ML4Science course.  
Feel free to open issues or request improvements!
