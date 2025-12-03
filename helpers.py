import os
import numpy as np
from scipy.stats import zscore
import scipy.io as sio
import matplotlib.pyplot as plt

def load_patient(patient_folder):
    epochs = []
    for fname in sorted(os.listdir(patient_folder)):
        if not fname.endswith(".npy"):
            continue
        fullpath = os.path.join(patient_folder, fname)
        try:
            data = np.load(fullpath)
            epoch = data.T          # the .npy files are saved as (30000,62)
            epochs.append(epoch)
        except Exception as e:
            print(f"Error loading {fullpath}: {e}")
    concatenated_epochs = np.concatenate(epochs, axis=1)
    return concatenated_epochs

def load_all_patients(root):
    patients_data = {}

    for patient in sorted(os.listdir(root)):
        name = os.path.basename(patient)
        patient_path = os.path.join(root, name)
        internal_path_name = "scout_data"
        patient_path = os.path.join(patient_path, internal_path_name)

        # Skip files, take only folders
        if os.path.isdir(patient_path):
                print(f"Loading {patient} ...")
                patients_data[name] = load_patient(patient_path)

    return patients_data

def sanity_check(patients_data, epoch_length_samples=30000):
    """
    Sanity check for patient data.

    Checks:
        - Total number of timepoints is divisible by epoch length
        - No region contains only zeros
    """
    issues = {}

    for patient, data in patients_data.items():
        patient_issues = []

        n_regions, n_samples = data.shape

        # Check for regions with all zeros
        zero_regions = np.where(np.all(data == 0, axis=1))[0]
        if len(zero_regions) > 0:
            patient_issues.append(f"Regions {zero_regions.tolist()} contain all zeros")

        # Check if total samples divisible by epoch_length
        if n_samples % epoch_length_samples != 0:
            patient_issues.append(f"Total samples {n_samples} not divisible by epoch length {epoch_length_samples}")

        if patient_issues:
            issues[patient] = patient_issues

    return issues

def threshold_mat(data, thresh):
    """Binarize a matrix (regions x time) according to a z-threshold."""
    return np.where(np.abs(data) > thresh, 1, 0)

def active_bin_times(binarized_signal, bin_size):
    """
    binarized_signal: (n_regions, n_samples)
    bin_size: number of timepoints per bin

    output:
        time_bins: (n_bins,) -> 1 if at least one region is active in the bin
    """
    n_regions, n_samples = binarized_signal.shape
    n_bins = n_samples // bin_size

    if n_bins == 0:
        return np.zeros(0, dtype=int)

    # truncate signal to have a number of samples multiple of bin_size
    trimmed = binarized_signal[:, :n_bins * bin_size]

    # reshape: (n_regions, n_bins, bin_size) --> splits the signal in bins
    reshaped = trimmed.reshape(n_regions, n_bins, bin_size)

    # 1) all over bin_size --> one if ANY the 20 timepoints in the bin are 1
    regions_active_in_bin = reshaped.any(axis=2).astype(int) # (n_regions, n_bins)

    return regions_active_in_bin

def find_avalanches(binned, min_duration):
    """
    binned: (n_regions, n_bins), binary
    return: list of dict, each with:
        - 'activity': (n_regions, n_bins_avalanche)
        - 'start_bin', 'end_bin'
    """
    n_bins = binned.shape[1]
    active_per_bin = binned.sum(axis=0)  # (n_bins,) number of regions over threshold per bin

    avalanches = []
    in_aval = False # initialize as out of an avalanche
    start = 0

    for t in range(n_bins):
        if active_per_bin[t] > 0 and not in_aval: # a new avalanche starts if at least one region is over threshold
            in_aval = True
            start = t
        elif active_per_bin[t] == 0 and in_aval: # the avalanche ends
            end = t - 1
            activity = binned[:, start:end+1]
            if (end - start + 1) >= min_duration:
                avalanches.append({
                    "activity": activity,
                    "start_bin": start,
                    "end_bin": end
                })
            in_aval = False 

    # if the signal ends with an ongoing avalanche
    if in_aval:
        end = n_bins - 1
        activity = binned[:, start:end+1]
        avalanches.append({
            "activity": activity,
            "start_bin": start,
            "end_bin": end
        })

    return avalanches

def compute_avalanche_features(avalanches):
    sizes = []
    durations = []
    branching_ns = [] 

    for aval in avalanches:
        activity = aval["activity"]  # shape: (n_regions, n_bins_av)
        n_bins_av = activity.shape[1]

        # size: total sum of 1 over time-bins and regions of a specific detected avalanche
        size = activity.sum()
        sizes.append(size)

        # duration: numbero of bins
        durations.append(n_bins_av)

        # branching: n(t+1)/n(t)
        branching_i=[]
        n_t = activity.sum(axis=0)  # number of regions active in a bin where an avalanche was found (n_bins_av,)
        for t in range(n_bins_av - 1):
            branching_i.append(n_t[t+1] / n_t[t])

    features = {}
    if len(sizes) > 0:
        sizes = np.array(sizes)
        durations = np.array(durations)
        branching_i = np.array(branching_i) if len(branching_i) > 0 else np.array([np.nan])

        features["mean_size"] = sizes.mean()
        features["max_size"] = sizes.max()
        features["mean_duration"] = durations.mean()
        features["max_duration"] = durations.max()
        features["branching_factor"] = np.nanmean(branching_i)
        print(features["branching_factor"])
    else:
        # if no avalanches have been found
        features["mean_size"] = 0.0
        features["max_size"] = 0.0
        features["mean_duration"] = 0.0
        features["max_duration"] = 0.0
        features["branching_factor"] = np.nan

    return features

def compute_ATM(avalanches, n_regions = 62):
    ATMs = []

    for aval in avalanches:
        counts = np.zeros((n_regions, n_regions), dtype=float)  # reset at every avalanche
        activity = aval["activity"]  # (n_regions, n_bins_av)
        n_bins_av = activity.shape[1]

        for t in range(n_bins_av - 1):
            active_now = np.where(activity[:, t] > 0)[0]
            active_next = np.where(activity[:, t+1] > 0)[0]
            for i in active_now:
                for j in active_next:
                    counts[i, j] += 1 

        # normalize to get the probability value
        row_sums = counts.sum(axis=1, keepdims=True)
        ATM = np.divide(counts, row_sums, out=np.zeros_like(counts), where=(row_sums != 0))
        ATMs.append(ATM)

    ATM_mean = np.mean(ATMs, axis=0)    # mean over all avalanches
    return ATM_mean


def save_atm_plot(atm_matrix, patient_id, out_folder="atm_plots"):
    """
    Save a heatmap of the ATM matrix for a patient without displaying it.
    """
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"{patient_id}_ATM.png")

    plt.figure(figsize=(8, 6))
    plt.imshow(atm_matrix, cmap="viridis", aspect="auto")
    plt.title(f"ATM - {patient_id}")
    plt.colorbar(label="Transition Probability")
    plt.xlabel("Region j")
    plt.ylabel("Region i")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close() 

    print(f"Saved ATM plot for {patient_id} → {save_path}")



def build_feature_matrix(patient_atm):
    """
    Convert patient-level ATM dict to a feature matrix X (patients x features),
    using all the directional couples (i → j)
    
    Parameters:
        patient_atm: dict {patient_id: ATM_matrix (n_regions x n_regions)}
        
    Returns:
        X: numpy array of shape (n_patients, n_regions * n_regions)
        patients: list of patient IDs in the same order as the rows of X
    """
    # Sort patient IDs to ensure consistent order
    patients = sorted(patient_atm.keys())

    patients = list(patient_atm.keys())
    first = patient_atm[patients[0]]
    n_regions = first.shape[0]
    n_features = n_regions * n_regions
    X = np.zeros((len(patients), n_features))
    for i, p in enumerate(patients):
        atm = patient_atm[p]
        # flatten the atm matrix
        X[i, :] = atm.reshape(-1)

    return X, patients

def process_group(root_folder, group_label, fs, bin_size, z_thresh, n_regions):
    patients = load_all_patients(root_folder)  # dict {patient_id: epochs_concatenated}
    sanity_check(patients)

    patient_features = {}  # save feature vectors here
    patient_atm = {}       # save ATMs here

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
        patient_features[patient] =  avalanche_features

        # Compute ATM
        transition_matrix = compute_ATM(avalanches, n_regions)
        patient_atm[patient] = transition_matrix
        save_atm_plot(transition_matrix, patient, out_folder="atm_plots")

    return patient_features, patient_atm