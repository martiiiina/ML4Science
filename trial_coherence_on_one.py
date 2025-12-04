import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_connectivity import spectral_connectivity_time
from helpers import save_mat_plot

# Path to the dataset
sub_path = r"Z:\acutestroke_data_combineflipping_final\unflipped_leftlesion\TiMeS_WP11_017\scout_data"
epochs = []

# Load data from files and concatenate epochs
for file in sorted(os.listdir(sub_path)):
    if not file.endswith(".npy"):
        continue
    full_path = os.path.join(sub_path, file)
    data = np.load(full_path)
    epoch = data.T  # Transpose data to match shape (n_channels, n_samples)
    epochs.append(epoch)
epochs = np.array(epochs)
print(f"Shape of epochs: {epochs.shape}")  # (n_epochs, n_channels, n_samples)

n_channels = epochs.shape[1]  # Number of regions/channels
ch_names = ['Region_' + str(i + 1) for i in range(n_channels)]  # Channel names

sfreq = 5000  # Sampling frequency (adjust this if needed)
n_times = epochs.shape[2]
n_epochs = epochs.shape[0]

# Create MNE info object
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
# Create the MNE EpochsArray object
epochs_array = mne.EpochsArray(epochs, info)

# Provide the freq points
min_freq = 8.0
max_freq = 12.0
freqs = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 4 + 1))

# Compute connectivity over time
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

con_mat = con_time.get_data(output="dense")
print(con_mat.shape)
con_mat_squeezed = np.squeeze(con_mat)
print(con_mat_squeezed.shape)

full_coh = np.tril(con_mat_squeezed) + np.tril(con_mat_squeezed, -1).T

save_mat_plot(full_coh, "trial", out_folder="trial_coh")