import os
import pandas as pd
import shutil
from datetime import datetime, timedelta

# User inputs
direc = '/Users/clevenger/Projects/data_assimilation2/test_dates/02162023/data_product_outputs/asi_spectral_inversion/6UT/'
outdir = '/Users/clevenger/Projects/data_assimilation2/test_dates/02162023/data_inputs/asi/lompe/'
date = '20230216'
time_csv = os.path.join(direc, 'time_ranges.csv')

start_time = '060000'  # Format: HHMMSS
end_time = '065959'    # Format: HHMMSS

# Helper function to convert time to seconds
def time_to_seconds(time_str):
    t = datetime.strptime(time_str, '%H%M%S')
    return t.hour * 3600 + t.minute * 60 + t.second

# Calculate the total seconds between start and end time
start_seconds = time_to_seconds(start_time)
end_seconds = time_to_seconds(end_time)
total_seconds = end_seconds - start_seconds

# Get all 'group_<number>' files
files = sorted([f for f in os.listdir(direc) if f.startswith('group_')])

# Number of groups
num_groups = len(files)

# Calculate cadence in seconds
cadence_seconds = total_seconds // num_groups  # Integer division to get the smallest interval

# Create output directory if it doesn't exist
os.makedirs(outdir, exist_ok=True)

# Starting datetime object based on the input date and start time
start_datetime = datetime.strptime(date + start_time, '%Y%m%d%H%M%S')

# Loop through each file and rename
for i, file in enumerate(files):
    # Calculate new timestamp for the file
    file_time = start_datetime + timedelta(seconds=i * cadence_seconds)
    
    # Construct new filename in 'YYYYMMDD_HHMMSS' format
    new_name = file_time.strftime('%Y%m%d_%H%M%S') + '.h5'
    
    # Full paths for source and destination
    src_file = os.path.join(direc, file)
    dest_file = os.path.join(outdir, new_name)
    
    # Copy and rename the file
    shutil.copy(src_file, dest_file)
    print(f"Copied and renamed {file} to {new_name}")

print("File copying and renaming complete.")
