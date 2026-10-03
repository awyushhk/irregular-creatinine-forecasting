import pandas as pd
import numpy as np

# 1 & 2. Inspect d_labitems.csv and find creatinine
d_labitems_path = r"d:\CKD mini project\renal-twin\data\mimic-iv-clinical-database-demo-2.2\hosp\d_labitems.csv"
labitems = pd.read_csv(d_labitems_path)
creatinine_items = labitems[labitems['label'].str.contains('creatinine', case=False, na=False)]
print("Found Creatinine Lab Items:")
print(creatinine_items[['itemid', 'label', 'fluid', 'category']])

# Using the most common blood creatinine itemid: 50912 (Blood)
# Let's verify and just use it if it's there
blood_creat_itemid = 50912
if blood_creat_itemid not in creatinine_items['itemid'].values:
    blood_creat_itemid = creatinine_items.iloc[0]['itemid']
print(f"\nSelected Item ID for Creatinine: {blood_creat_itemid}")

# 3 & 4. Inspect labevents.csv and extract columns
labevents_path = r"d:\CKD mini project\renal-twin\data\mimic-iv-clinical-database-demo-2.2\hosp\labevents.csv"
# We can read in chunks or specify columns to save memory, though demo dataset is small
labevents = pd.read_csv(labevents_path, usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'])
creat_events = labevents[labevents['itemid'] == blood_creat_itemid].copy()

print(f"\nInitial creatinine observations: {len(creat_events)}")

# 5. Remove invalid/non-numeric values
creat_events = creat_events.dropna(subset=['valuenum'])
print(f"After removing NaN values: {len(creat_events)}")

# Convert charttime to datetime
creat_events['charttime'] = pd.to_datetime(creat_events['charttime'])

# 6. Sort chronologically for every patient
creat_events = creat_events.sort_values(by=['subject_id', 'charttime'])

# 7. Keep patients having at least 3 creatinine measurements
patient_counts = creat_events['subject_id'].value_counts()
valid_patients = patient_counts[patient_counts >= 3].index
final_df = creat_events[creat_events['subject_id'].isin(valid_patients)].copy()

print(f"After filtering for >= 3 measurements: {len(final_df)} observations")

# 8. Report metrics
num_patients = final_df['subject_id'].nunique()
num_observations = len(final_df)

# Calculate measurement intervals per patient
final_df['prev_time'] = final_df.groupby('subject_id')['charttime'].shift(1)
final_df['interval_hours'] = (final_df['charttime'] - final_df['prev_time']).dt.total_seconds() / 3600.0

intervals = final_df['interval_hours'].dropna()

median_interval = intervals.median()
mean_interval = intervals.mean()
max_interval = intervals.max()

min_creat = final_df['valuenum'].min()
max_creat = final_df['valuenum'].max()

print("\n--- Summary Statistics ---")
print(f"Number of patients: {num_patients}")
print(f"Number of creatinine observations: {num_observations}")
print(f"Median measurement interval: {median_interval:.2f} hours")
print(f"Mean measurement interval: {mean_interval:.2f} hours")
print(f"Maximum measurement interval: {max_interval:.2f} hours")
print(f"Minimum creatinine: {min_creat}")
print(f"Maximum creatinine: {max_creat}")
