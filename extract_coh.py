import numpy as np
import mne
import pandas as pd
from scipy.stats import zscore
from mne_connectivity import spectral_connectivity_time
from helpers import *

# 1. Load data
root = r"path/to/clean"
patients = load_all_patients_coh(root)  # dictionary, k: patients_id, v: np.array of non-concatenated epochs


rows = []

for patient, epochs in patients.items():
    print(epochs.shape)
    n_epochs = epochs.shape[0]
    n_regions = epochs.shape[1]
    print(f"Processing patient {patient} ({n_epochs} epochs)")

    # Create MNE object
    ch_names = ['Region_' + str(i + 1) for i in range(n_regions)]  # Channel names
    sfreq = 5000
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    epochs_array = mne.EpochsArray(epochs, info)
    
    # Compute connectivity over time (default: alpha band)
    min_freq = 8.0
    max_freq = 12.0
    freqs = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 4 + 1))

    con_time = spectral_connectivity_time(
        epochs_array,
        freqs,
        method="coh",
        average=True,
        sfreq=sfreq,
        mode="multitaper",
        fmin=min_freq,
        fmax=max_freq,
        faverage=True,
    )

    # Get the coherence matrix
    con_mat = np.squeeze(con_time.get_data(output="dense"))  # (62,62)
    # Save the coherence matrix 
    full_coh = np.tril(con_mat) + np.tril(con_mat, -1).T
    save_mat_plot(full_coh, patient, out_folder="coh_plots")

    # Flatten only the lower triangular part (excluding diagonal)
    lower_triangular_indices = np.tril_indices(n_regions, k=-1)  # k=-1 excludes diagonal
    flattened_coh = con_mat[lower_triangular_indices]  # Extract lower triangular part
    print(flattened_coh.shape)

    # Z-score
    flattened_coh_z = zscore(flattened_coh)

    # Build row
    row = {
        "patient_id": patient,
        "label": label_from_patient_id(patient)
    }
    # Add coherence features as separate columns
    for i, v in enumerate(flattened_coh_z):
        row[f"coh_{i}"] = v
    rows.append(row)

# Build final dataframe
df = pd.DataFrame(rows)
df.to_csv("coherence_dataset.csv", index=False)

print("Saved coherence_dataset.csv")