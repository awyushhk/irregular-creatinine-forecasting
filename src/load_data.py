import pandas as pd
import numpy as np
import os

def load_and_filter_mimic_data(data_dir=r"d:\CKD mini project\renal-twin\data\mimic-iv-clinical-database-demo-2.2"):
    labitems_path = os.path.join(data_dir, "hosp", "d_labitems.csv")
    labevents_path = os.path.join(data_dir, "hosp", "labevents.csv")
    
    # We already know from exploration that 50912 is Blood Creatinine
    blood_creat_itemid = 50912
    
    labevents = pd.read_csv(labevents_path, usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'])
    creat_events = labevents[labevents['itemid'] == blood_creat_itemid].copy()
    
    creat_events = creat_events.dropna(subset=['valuenum'])
    creat_events['charttime'] = pd.to_datetime(creat_events['charttime'])
    creat_events = creat_events.sort_values(by=['subject_id', 'charttime'])
    
    # Keep patients having at least 3 creatinine measurements
    patient_counts = creat_events['subject_id'].value_counts()
    valid_patients = patient_counts[patient_counts >= 3].index
    final_df = creat_events[creat_events['subject_id'].isin(valid_patients)].copy()
    
    return final_df
