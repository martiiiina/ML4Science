import os 
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root = "/Volumes/MB_fall_2025//NeuronalAvalanches_dataset/acutestroke_data_combineflipping_final/flipped_rightlesion"

def load_component(mat_path):
    data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    times = data["Time"]
    values = data["Value"]
    atlas = data["Atlas"]

    times = np.ravel(times)
    values = np.array(values)

    return {"Time": times, "Value": values, "Atlas": atlas}

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
