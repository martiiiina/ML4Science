# ML4Science: Avalanche and ATM Analysis

This repository contains code for preprocessing, feature extraction, and analysis of patient neural recordings. The focus is on detecting **neuronal avalanches** and computing **Avalanche Transition Matrices (ATMs)** from multi-region EEG time-series.

## Project Structure

```
ML4Science/
├── Literature/                              # Reference papers
├── atm_plots*/                              # Heatmaps of ATM matrices for each patient
├── coh_plots*/                              # Heatmaps of Coh matrices for each patient
├── dataset_info/                            # Information on data acquisition
├── shap_plots/                              # Shap values plots
├── shap_values/                             # Shap values stored in .csv files for all seeds
├── xgb__best_models/                        # Best XGB model for each random state, used for SHAP analysis
├── README.md                                # Project description and instructions
├── atm_dataset*.csv                         # DataFrame of ATM features
├── coherence_dataset*.csv                   # DataFrame of Coh features
├── extract_atm.py                           # Extract ATMs and save features in a .csv file
├── extract_coh.py                           # Extract Cohs and save features in a .csv file
├── helpers.py                               # Helper functions for loading, binarization, ATM/Coh computation
├── optimize_bf.py                           # Script for time binning optimization
├── results_atm.xlsx                         # Balanced accuracies from ATM classification for all seeds
├── results_coh.xlsx                         # Balanced accuracies from Coh classification for all seeds
├── rf_classifier.py                         # Random Forest classifier
├── shap_analysis.py                         # Script to perform SHAP analysis on XGB
├── shap_values_atm_regions_all_seeds.xlsx   # Summary of SHAP analysis
├── statistical_analysis.ipynb               # Statistial analysis on NA scalar features
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
- openpyxl

Install dependencies:

```
pip install -r requirements.txt
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
   - **Time-binning**: group time points into bins 
   - **Avalanche detection**: identify consecutive active bins across regions.
   - Compute **avalanche features**: mean size, max size, mean duration, max duration, branching factor.
   - Construct **Avalanche Transition Matrices (ATMs)**: probability of transitions between active regions in consecutive bins.
   - Save ATM heatmaps in `atm_plots/`.
   - Flatten ATMs into feature vectors and build a **patient-level dataframe** (`atm_dataset.csv`).

2. **Coherence Feature Extraction**  
   - Load patient EEG data as epochs without concatenation.
   - Create **MNE EpochsArray** objects for each patient.
   - Compute **spectral coherence** in a frequency band using `mne_connectivity.spectral_connectivity_time`.
   - Extract the **full coherence matrix**, symmetrize it, and save heatmaps in `coh_plots/`.
   - Flatten the **lower triangular part** of the coherence matrix to create feature vectors.
   - Apply **Z-score normalization** to coherence features.
   - Build a **patient-level dataframe** (`coherence_dataset.csv`).

3. **Branching factor optimization**
   - Computes branching factors for different candidate bin sizes
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
   - Best hyperparameters, CV and test balanced accuracy are saved to `results_atm.xlsx` or `results_coherence.xlsx`  
   - Balanced accuracies are saved for each seed in the `results*.xlsx`file
   - For multiple random seeds, a final summary with mean and standard deviation of test balanced accuracy is printed  

### Example Usage

To classify stroke vs. healthy EEG, you can select:
- The feature type:
   - Avalanche Transition Matrices (ATM)
   - Coherence
- The extracted dataset variant (minimum duration or frequency band)

1. Select the extracted dataset in the desired classification script (rf_classifier.py, svm_classifier.py, xgb_classifier.py)
- Modify the dataset filename according to the parameterization you want to use (line 19):
   - ATM datasets (different bin / duration parameters):
   ```python
      df = pd.read_csv("atm_dataset_20.csv")
      df = pd.read_csv("atm_dataset_30.csv")
      df = pd.read_csv("atm_dataset_40.csv")
   ```
   - Coherence datasets (frequency bands):
   ```python
   df = pd.read_csv("coherence_dataset_alpha.csv")
   df = pd.read_csv("coherence_dataset_beta.csv")
   ```

2. Select the feature type (line 20)
- Change the feature selection line to match the feature type:
  ```python
  X = df.filter(regex="^atm_").values  # For ATM features
  X = df.filter(regex="^coh_").values  # For Coherence features

3. Update the result filename accordingly
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
4. Run the classifier
- The chosen classifier is evaluated over 50 random seeds:
```
python svm_classifier.py
```

## SHAP Stability Analysis Across Random Seeds

This pipeline evaluates the **stability and robustness of SHAP explanations** for XGBoost models trained on ATM connectivity features by comparing results across up to **50 random seeds**. The goal is to ensure that reported feature and brain-region importances are **not driven by random splits**, but are consistently supported across models.

### Data and Models

- **Dataset**: `atm_dataset.csv`  
  - Features: columns starting with `atm_` (flattened 62 × 62 ATM connectivity)  
  - Label: `label`

- **Models**: `xgb_best_models/best_xgb_model_seed_{seed}.pkl`  
  - One independently trained XGBoost pipeline per seed  
  - Each pipeline includes a scaler and an XGBoost classifier

- **Sampling**:  
  - Up to 100 samples are randomly selected  
  - The same samples are used for all seeds to allow fair SHAP comparison

### Analysis Workflow

1. **SHAP computation**  
   - SHAP values are computed for each seed using `TreeExplainer`  
   - Results are stored in a 3D array:  
     `(n_seeds, n_samples, n_features)`

2. **Feature stability across seeds**  
   - Mean and standard deviation of absolute SHAP values are computed per feature  

3. **SHAP visualizations**  
   - Beeswarm plots are generated for each seed  
   - Aggregated beeswarm and bar plots summarize global, seed-robust feature importance

4. **ROI-level reconstruction**  
   - Features are reshaped into 62 × 62 ROI–ROI matrices  
   - Mean absolute SHAP values are computed per connection and per seed

5. **Multi-seed aggregation**  
   - ROI–ROI SHAP values are averaged across seeds  
   - Node-level (ROI) importance is derived by pooling all connections involving each region

6. **Heatmap visualization**  
   - A 62 × 62 heatmap shows aggregated SHAP importance for ATM connections across all seeds

### Outputs

- **Plots** (`shap_plots/`)
  - Feature importance distribution across seeds  
  - Per-seed and aggregated beeswarm plots  
  - Aggregated ROI–ROI SHAP heatmap
  - Beeswarm per MAV of SHAP values
  - Bar plot per MAV of SHAP values 

- **Tables** (`shap_values/`)
  - Per-seed ROI–ROI SHAP rankings (CSV)  
  - Combined CSV and Excel file with:
    - Seed-wise summaries
    - Aggregated ROI–ROI importance
    - Node (ROI) importance across all seeds

### Interpretation

- **Consistent high SHAP values across seeds** → robust features or connections  
- **Low cross-seed variance** → stable explanations  
- **High node importance** → brain regions acting as influential hubs

## Author

Project maintained by **Martina, Clotilde and Martina** for the EPFL ML4Science course.  
Feel free to open issues or request improvements!