import os 
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root = "/Volumes/MB_fall_2025//NeuronalAvalanches_dataset/acutestroke_data_combineflipping_final/flipped_rightlesion"

def load_component(mat_path):
    # più robusto: squeeze structs e cerca nomi alternativi / inferisce variabili
    data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    # possibili nomi
    time_candidates = ["Times", "times", "Time", "time", "t", "T"]
    value_candidates = ["Values", "values", "Data", "data", "X", "x", "signals", "Signals"]
    atlas_candidates = ["Atlas", "atlas"]

    def pick_key(cands):
        for k in cands:
            if k in data:
                return data[k]
        return None

    times = pick_key(time_candidates)
    values = pick_key(value_candidates)
    atlas = pick_key(atlas_candidates)

    # fallback: cerca un vettore 1D lungo per Times
    if times is None:
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 1 and v.size > 10:
                    times = v
                    break

    # fallback: cerca una matrice 2D con molte colonne per Values
    if values is None:
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                # probabilmente shape = (n_regions, n_samples)
                if max(v.shape) > 100:
                    values = v
                    break

    if times is None:
        raise KeyError(f"'Times' not found in {mat_path}")
    if values is None:
        raise KeyError(f"'Values' not found in {mat_path}")

    times = np.ravel(times)
    values = np.array(values)

    return {"Times": times, "Values": values, "Atlas": atlas}

def load_patient(patient_folder):
    components = []

    # os.walk va in tutte le sotto-cartelle
    for root_dir, dirs, files in os.walk(patient_folder):
        for fname in files:
            if not fname.endswith(".mat"):
                continue
            fullpath = os.path.join(root_dir, fname)
            try:
                comp = load_component(fullpath)
                components.append(comp)
            except Exception as e:
                print(f"Errore caricando {fullpath}: {e}")

    return components

def load_all_patients(root):
    patients_data = {}

    for patient in sorted(os.listdir(root)):
        name = os.path.basename(patient)
        patient_path = os.path.join(root, name)
        to_add = "_T1_RS_Eyes_Open_6_ICAclean"
        internal_path_name = name + to_add
        patient_path = os.path.join(patient_path, internal_path_name)

        # Skip files, take only folders
        if os.path.isdir(patient_path):
            print(f"Loading {patient} ...")
            patients_data[internal_path_name] = load_patient(patient_path)

    return patients_data

patients = load_all_patients(root)

print(patients.keys())
p = patients["TiMeS_WP11_001_T1_RS_Eyes_Open_6_ICAclean"] # lista di componenti ICA
print(len(p))  
print(type(p[0]))
print(p[0].keys())
print("Times key present?", "Times" in p[0])
T = p[0].get("Times")