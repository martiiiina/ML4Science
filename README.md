"""
Acute Stroke EEG Dataset (NumPy Format)
---------------------------------------

This dataset contains EEG recordings from acute stroke patients. All recordings
were originally stored as MATLAB .mat files and have been converted to .npy
format for fast loading in Python.

Data Format
-----------
Each EEG epoch is stored as a separate .npy file with the following properties:
- Shape: (62, 30000)
- Cortical regions: 62 (DKT atlas)
- Sampling rate: 5000 Hz
- Duration: 6 seconds
- Data type: float NumPy array

Folder Structure
----------------
root/
    TiMeS_WP11_001/
        TiMeS_WP11_001_T1_RS_Eyes_Open_6_ICAclean/
            npy_data/
                TiMeS_WP11_001_T1_RS_Eyes_Open_6_ICAclean_1_DKT_mean.npy
                TiMeS_WP11_001_T1_RS_Eyes_Open_6_ICAclean_2_DKT_mean.npy
                ...

    TiMeS_WP11_002/
        TiMeS_WP11_002_T1_RS_Eyes_Open_6_ICAclean/
            npy_data/
                TiMeS_WP11_002_T1_RS_Eyes_Open_6_ICAclean_1_DKT_mean.npy
                ...

Summary:
- One folder per patient: TiMeS_WP11_XXX
- Internal EEG folder: {TiMeS_WP11_XXX}_T1_RS_Eyes_Open_6_ICAclean
- All .npy epoch files are stored in npy_data/
- Each .npy file corresponds to one EEG epoch

Loading Data in Python
----------------------
Use load_patient(npy_folder) to load all epochs for a single patient.

Input:
    Path to the patient's npy_data/ folder
Output:
    List of NumPy arrays (each of shape (62, 30000))

Example usage:

    root = r"Z:\\acutestroke_data_combineflipping_final\\flipped_rightlesion"

    from helpers import load_all_patients

    patients = load_all_patients(root)

"""