import numpy as np
from scipy.stats import zscore
from helpers import *

# 1. Load data
root = "/Volumes/MB_fall_2025//NeuronalAvalanches_dataset/acutestroke_data_combineflipping_final/flipped_rightlesion"
patients = load_all_patients(root)  # dictionary, k: patients_id, v: np.array of concatenated epochs
sanity_check(patients)

# 2. Signal binarization and anavalnches construction
fs = 5000
bin_size = int(0.004 * fs)       # 20 samples for a binning of 4 ms
z_thresh= 2
n_regions = 62
patient_atm = {}

for patient, times in patients.items():
    n_epochs = times.shape[1] // 30000  
    print(f"Processing patient {patient} ({n_epochs} epochs)")

    # Z-score normalization (along time)
    z_values = zscore(times, axis=1)

    # Binarize according to z_thresh
    binarized_signal = threshold_mat(z_values, z_thresh)

    # Divide the signal in time bins 
    binned_signal = active_bin_times(binarized_signal, bin_size)

    # Find avalanches 
    avalanches = find_avalanches(binned_signal, min_duration=3)
    print(f"{len(avalanches)} avalanches found in the binned signal")

    # Compute avalanches features
    avalanche_features = compute_avalanche_features(avalanches)

    # Compute ATM
    transition_matrix = compute_ATM(avalanches, n_regions)
    patient_atm[patient] = transition_matrix
    save_atm_plot(transition_matrix, patient, out_folder="atm_plots")

# 3. Build feature matrix
X, patient_ids = build_feature_matrix(patient_atm)

# 3. Build y labels
#y = np.array([labels_patient[pid] for pid in patient_ids])


