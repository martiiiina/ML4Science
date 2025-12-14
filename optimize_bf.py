# This script processes multi-region EEG data from multiple patients to analyze neuronal avalanches.
# The key steps are:
# 1. **Loading EEG data**: Patient data is loaded from a specified directory and structured as a dictionary of concatenated epochs.
# 2. **Signal Binarization**: The EEG signals are Z-score normalized across time and then binarized using a threshold.
# 3. **Avalanche Construction**: The signal is divided into time bins, and neuronal avalanches are detected from the binned signal.
# 4. **Avalanche Feature Calculation**: The branching factor of each detected avalanche is calculated, and the results are compared across different time bin sizes.
# 5. **Normalization**: A norm is calculated between branching factors and a reference value (ones vector).
#
# The output is a list of norm values representing the branching factor discrepancies across the different time bin sizes for each patient.


from scipy.stats import zscore
from helpers import *

# 1. Load data
root = r"\\sv-nas1.rcp.epfl.ch\Hummel-Data\TiMeS\Students_Interns\MB_fall_2025\NeuronalAvalanches_dataset"
patients = load_all_patients(root)  # dictionary, k: patients_id, v: np.array of concatenated epochs

# 2. Signal binarization and avalanches construction
fs = 5000
bin_size = [1, 3, 5, 7]       
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