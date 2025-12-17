import pandas as pd
from scipy.stats import zscore
from helpers import *

# TODO: NA have fat-tailed distributions in terms of length, plot for the report and to justify the binning

# 1. Load data
root = r"\\sv-nas1.rcp.epfl.ch\Hummel-Data\TiMeS\Students_Interns\MB_fall_2025\NeuronalAvalanches_dataset"
patients = load_all_patients(root)  # dictionary, k: patients_id, v: np.array of concatenated epochs

# 2. Signal binarization and avalanches construction
fs = 5000
bin_size = int(0.0002 * fs)       # 20 samples for a binning of 4 ms
z_thresh = 2.5
n_regions = 62
rows = []

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
    avalanches = find_avalanches(binned_signal, min_duration=40)
    print(f"{len(avalanches)} avalanches found in the binned signal")

    durations = [a["activity"].shape[1] for a in avalanches]
    print("   Mean duration:", np.mean(durations))
    print("   Median duration:", np.median(durations))
    vals, counts = np.unique(durations, return_counts=True)
    print("   Duration distribution:")
    for v, c in zip(vals, counts):
        print(f"      length {v}: {c} avalanches")

    # Compute avalanches features
    avalanche_features = compute_avalanche_features(avalanches)

    # Compute ATM
    transition_matrix = compute_ATM(avalanches, n_regions)
    save_mat_plot(transition_matrix, patient, out_folder="atm_plots")

    flattened_atm = transition_matrix.flatten()     # shape (3844,)
    row = {
        "patient_id": patient,
        "label": label_from_patient_id(patient)
    }
    # Add coherence features as separate columns
    for i, v in enumerate(flattened_atm):
        row[f"atm_{i}"] = v
    rows.append(row)

# Build final dataframe
df = pd.DataFrame(rows)
df.to_csv("atm_dataset.csv", index=False)

print("Saved atm_dataset.csv")