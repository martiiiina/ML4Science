import numpy as np
import mne
from mne_connectivity import spectral_connectivity_time
from helpers import *

# 1. Load data
root = r"\\sv-nas1.rcp.epfl.ch\Hummel-Data\TiMeS\Students_Interns\MB_fall_2025\NeuronalAvalanches_dataset"
patients = load_all_patients(root)  # dictionary, k: patients_id, v: np.array of concatenated epochs
sanity_check(patients)

for patient, patient_data in patients.items():
    epochs = patient_data["epochs"]
    print(epochs.shape)
    concatenated_epochs = patient_data["concatenated_epochs"]
    print(concatenated_epochs.shape)
    n_samples = 30000
    n_epochs = concatenated_epochs.shape[1] // n_samples
    n_regions = concatenated_epochs.shape[0]
    print(f"Processing patient {patient} ({n_epochs} epochs)")

    
    ch_names = ['Region_' + str(i + 1) for i in range(n_regions)]  # Channel names

    sfreq = 5000

    # Create MNE info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    # Create the MNE EpochsArray object
    epochs_array = mne.EpochsArray(epochs, info)

    # We will try two different connectivity measurements as an example
    connectivity_methods = ["coh"]
    n_con_methods = len(connectivity_methods)

    Freq_Bands = {"broad": [8.0, 12.0]}
    n_freq_bands = len(Freq_Bands)
    min_freq = np.min(list(Freq_Bands.values()))
    max_freq = np.max(list(Freq_Bands.values()))
    # Provide the freq points
    freqs = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 4 + 1))
    # The dictionary with frequencies are converted to tuples for the function
    fmin = tuple([list(Freq_Bands.values())[f][0] for f in range(len(Freq_Bands))])
    fmax = tuple([list(Freq_Bands.values())[f][1] for f in range(len(Freq_Bands))])


    # Pre-allocatate memory for the connectivity matrices
    con_time_array = np.zeros(
        (n_con_methods, n_epochs, n_regions, n_regions, n_freq_bands)
    )
    con_time_array[con_time_array == 0] = np.nan  # nan matrix


    # Compute connectivity over time
    con_time = spectral_connectivity_time(
        epochs_array,
        freqs,
        method=connectivity_methods,
        average=True,
        sfreq=sfreq,
        mode="multitaper",
        fmin=fmin,
        fmax=fmax,
        faverage=True,
    )

    con_mat = con_time.get_data(output="dense")
    print(con_mat.shape)
    con_mat_squeezed = np.squeeze(con_mat)
    print(con_mat_squeezed.shape)

    full_coh = np.tril(con_mat_squeezed) + np.tril(con_mat_squeezed, -1).T

    save_mat_plot(full_coh, "trial", out_folder="trial_coh")

