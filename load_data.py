import os 
import scipy.io as sio
import numpy as np

root = "/Volumes/MB_fall_2025//NeuronalAvalanches_dataset/acutestroke_data_combineflipping_final/flipped_rightlesion"

def load_component(mat_path):
    data = sio.loadmat(mat_path)
    return {
        "Times": data["Times"].flatten(),        # 1 × 30000
        "Values": data["Values"],                # 62 × 30000
        "Atlas": data["Atlas"] 
    }

def load_patient(patient_folder):
    components = []

    # os.walk va in tutte le sotto-cartelle
    for root, dirs, files in os.walk(patient_folder):
        for fname in files:
            if fname.endswith(".mat"):
                fullpath = os.path.join(root, fname)
                # qui puoi usare direttamente sio.loadmat
                comp = sio.loadmat(fullpath)
                components.append(comp)

    return components

def load_all_patients(root):
    patients_data = {}

    for patient in sorted(os.listdir(root)):
        patient_path = os.path.join(root, patient)
        name = os.path.basename(patient)
        patient_path = os.path.join(root, name)
        to_add = "_T1_RS_Eyes_Open_6_ICAclean"
        internal_path_name = name + to_add
        patient_path = os.path.join(patient_path, internal_path_name)
        print(patient_path)

        # Skip files, take only folders
        if os.path.isdir(patient_path):
            print(f"Loading {patient} ...")
            patients_data[patient] = load_patient(patient_path)

    return patients_data

patients = load_all_patients(root)
