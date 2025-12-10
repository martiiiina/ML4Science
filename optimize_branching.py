import pandas as pd
from scipy.stats import zscore
from helpers import *

# 1. Load data
root = r"\\sv-nas1.rcp.epfl.ch\Hummel-Data\TiMeS\Students_Interns\MB_fall_2025\NeuronalAvalanches_dataset"
#root = "/Volumes/MB_fall_2025//NeuronalAvalanches_dataset"
patients = load_all_patients(root)  # dictionary, k: patients_id, v: np.array of concatenated epochs

# 2. Signal binarization and avalanches construction
fs = 5000
bin_size = [1, 3, 5, 7]       # 20 samples for a binning of 4 ms
z_thresh = 2.5
n_regions = 62
rows = []
branchings = []
norms = []
ones = np.ones([68, 1])

for bin in bin_size:
    for patient, times in patients.items():
        n_epochs = times.shape[1] // 30000  
        print(f"Processing patient {patient} ({n_epochs} epochs)")

        # Z-score normalization (along time)
        z_values = zscore(times, axis=1)

        # Binarize according to z_thresh
        binarized_signal = threshold_mat(z_values, z_thresh)

        # Divide the signal in time bins 
        binned_signal = active_bin_times(binarized_signal, bin)

        # Find avalanches 
        avalanches = find_avalanches(binned_signal, min_duration=1)
        print(f"{len(avalanches)} avalanches found in the binned signal")

        # Compute avalanches features
        avalanche_features = compute_avalanche_features(avalanches)

        branching = avalanche_features["branching_factor"]
        branchings.append(branching)
    
    diff = branchings - ones
    norm = np.linalg.norm(diff)
    norms.append(norm)

print(norms)