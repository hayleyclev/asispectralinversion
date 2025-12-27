import numpy as np
import h5py
import os
from PIL import Image
import glob
import re
import shutil
from bs4 import BeautifulSoup
import requests
from os.path import exists
import wget
from transformation import feed_data
import pandas as pd
from artifact_removing import remove_artifacts

"""
Purpose of this script:
    - handle input data
        - GLOW lookup table for a specific hour of an event
        - ASI PNGs from Don's archive (http://optics.gi.alaska.edu/amisr_archive/PKR/DASC/PNG/)
    - gets everything into a nice h5 format before preprocessing and inversion
    - contains main function call that feeds processed h5 files into inversion pipeline
"""

def parse_filename_timestamp(fname):
    """
    Extracts date, time, wavelength from filenames such as:
    PKR_YYYYMMDD_HHMMSS_WWWW.png
    """
    base = os.path.basename(fname)

    # Updated regex to match actual filename pattern
    m = re.match(r'PKR_(\d{8})_(\d{6})_(\d{4})\.png', base)
    if not m:
        raise ValueError(f"Filename does not match expected ASI format: {fname}")

    date = m.group(1)
    time = m.group(2)
    wavelength = m.group(3)

    return date, time, wavelength


def load_local_imagery(folder):
    """
    Loads PNGs from a directory. If PNGs are already sorted into wavelength
    subfolders (0428, 0558, 0630), the function simply returns.
    """

    print("Loading local PNG imagery from directory...")

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Input directory does not exist: {folder}")

    # Expected sorted folder layout
    expected_subfolders = {"0428", "0558", "0630"}
    existing_subfolders = {d for d in os.listdir(folder)
                           if os.path.isdir(os.path.join(folder, d))}

    # if already downloaded/sorted, shortcut out
    if expected_subfolders.issubset(existing_subfolders):
        print("Detected existing sorted wavelength folders — skipping PNG checks.")
        return folder

    # PNGs still need to be sorted
    pngs = sorted([f for f in os.listdir(folder) if f.endswith(".png")])

    if len(pngs) == 0:
        raise RuntimeError(f"No PNG files found in {folder} (and no sorted subfolders).")

    wavelengths_found = set()
    for f in pngs:
        _, _, wl = parse_filename_timestamp(f)
        wavelengths_found.add(wl)

    expected = {"0428", "0558", "0630"}
    if not expected.issubset(wavelengths_found):
        print("WARNING: Missing one or more expected wavelengths 0428, 0558, 0630")
        print("Found:", wavelengths_found)

    return folder


def sort_pngs(folder):
    """
    Moves PNG files into subfolders based on wavelength.
    If already sorted, does nothing.
    """

    print("Sorting PNGs...")

    expected_subfolders = {"0428", "0558", "0630"}
    existing_subfolders = {d for d in os.listdir(folder)
                           if os.path.isdir(os.path.join(folder, d))}

    # do nothing if already sorted
    top_pngs = [f for f in os.listdir(folder) if f.endswith(".png")]
    if expected_subfolders.issubset(existing_subfolders) and len(top_pngs) == 0:
        print("PNG files already sorted — continuing.")
        return folder

    # sort if not yet sorted
    pattern = re.compile(r"_(\d{4})\.png$")

    for file in top_pngs:
        match = pattern.search(file)
        if match:
            wl = match.group(1)
            dest = os.path.join(folder, wl)
            os.makedirs(dest, exist_ok=True)
            shutil.move(os.path.join(folder, file), os.path.join(dest, file))

    print("PNG sorting complete.")
    return folder


def png_2_h5(folder):
    """
    Purpose:
        - reads sorted PNG files
        - creates a new subfolder to store hdf5 files
        - converts each PNG file into an hdf5 file
        - stores converted files into new hdf5 folder
    """

    print("Converting PNGs to HDF5s...")
    
    lambdas = ['0630', '0558', '0428'] # red, green, blue wavelengths
    h5_dirs = [] # initalize empty directory

    # Go by each wavelength subfolder
    for lam in lambdas:
        lambda_folder = os.path.join(folder, lam)
        h5_folder = os.path.join(lambda_folder, 'h5_files')
        os.makedirs(h5_folder, exist_ok=True) # bypass if already exists, make if not
        h5_dirs.append(h5_folder)
        
        # Process PNG files
        for file in os.listdir(lambda_folder):
            if file.endswith('.png'): # pulls all files with PNG extension
                src_file = os.path.join(lambda_folder, file)
                h5_file = os.path.join(h5_folder, file.replace('.png', '.h5'))
                
                # Convert PNG to numpy array
                with Image.open(src_file) as img:
                    img_array = np.array(img)

                # Write numpy array to HDF5 file
                with h5py.File(h5_file, 'w') as h5:
                    h5.create_dataset('data', data=img_array)

    return folder


def group_by_timestamp(folder, base_outdir):
    """
    Groups frames strictly by timestamp order:
        group 1 = first 0428, first 0558, first 0630
        group 2 = second 0428, second 0558, second 0630
        and so on...

    Output file name: HHMMSSstart_HHMMSSend.h5

    Spreadsheet includes:
        - wavelengths
        - image filenames
        - timestamps
        - group window (start–end)
    """

    print("Grouping frames by aligned timestamps...")

    wavelengths = ["0428", "0558", "0630"]
    per_lambda_lists = {}

    # Load and sort the H5 files for each wavelength
    for wl in wavelengths:
        lam_folder = os.path.join(folder, wl, "h5_files")
        if not os.path.isdir(lam_folder):
            raise RuntimeError(f"Missing folder: {lam_folder}")

        h5_list = sorted(
            glob.glob(os.path.join(lam_folder, "*.h5")),
            key=lambda f: re.search(r'_(\d{6})_\d{4}\.h5$', f).group(1)
                         if re.search(r'_(\d{6})_\d{4}\.h5$', f) else ""
        )

        if len(h5_list) == 0:
            raise RuntimeError(f"No H5 files found in: {lam_folder}")

        per_lambda_lists[wl] = h5_list

    # enforce equal-length sequences
    min_len = min(len(per_lambda_lists[wl]) for wl in wavelengths)
    for wl in wavelengths:
        per_lambda_lists[wl] = per_lambda_lists[wl][:min_len]

    # grouping
    summary_rows = []
    os.makedirs(base_outdir, exist_ok=True)

    # Timestamp extractor
    def ts(fname):
        base = os.path.basename(fname)
        # main pattern: PKR_YYYYMMDD_HHMMSS_WWWW.h5
        m = re.match(r'PKR_\d{8}_(\d{6})_\d{4}\.h5$', base)
        if m:
            return m.group(1)
        # fallback pattern
        m = re.search(r'_(\d{6})_\d{4}\.h5$', base)
        if m:
            return m.group(1)
        raise ValueError(f"Cannot extract timestamp from filename: {fname}")

    for idx in range(min_len):

        # One image per wavelength
        files = {wl: per_lambda_lists[wl][idx] for wl in wavelengths}

        # Timestamps for these three files
        timestamps = {wl: ts(files[wl]) for wl in wavelengths}
        start_time = min(timestamps.values())

        # Compute end_time
        if idx < min_len - 1:
            # Next group's earliest timestamp
            next_times = [ts(per_lambda_lists[wl][idx + 1]) for wl in wavelengths]
            next_start = min(next_times)

            # end = one second before next_start
            hh = int(next_start[0:2])
            mm = int(next_start[2:4])
            ss = int(next_start[4:6])
            next_dt = hh * 3600 + mm * 60 + ss
            end_dt = max(0, next_dt - 1)
            end_time = f"{end_dt//3600:02d}{(end_dt%3600)//60:02d}{end_dt%60:02d}"

        else:
            # Last group ends at its own max timestamp
            end_time = max(timestamps.values())

        # Naming
        group_name = f"{start_time}_{end_time}"
        out_folder = os.path.join(base_outdir, group_name)
        os.makedirs(out_folder, exist_ok=True)

        # Create grouped H5 file
        out_h5 = os.path.join(out_folder, f"group_{group_name}.h5")
        with h5py.File(out_h5, "w") as h5out:
            for wl in wavelengths:
                with h5py.File(files[wl], "r") as h5src:
                    data = h5src["data"][:]
                    h5out.create_dataset(f"frame_{wl}", data=data)

        # spreadsheet rows
        for wl in wavelengths:
            summary_rows.append({
                "wavelength": wl,
                "h5_file": files[wl],
                "timestamp": timestamps[wl],
                "group_name": group_name,
                "window_start": start_time,
                "window_end": end_time
            })

    Write spreadsheet
    df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(base_outdir, "group_timing_summary.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n✔ Grouping complete.")
    print(f"✔ Spreadsheet written: {out_csv}")
    return folder

    
def make_time_list_sliding_window(folder, starttime, endtime, window_size=3):
    # convert to datetime for easier comparison
    start_dt = datetime.strptime(starttime, "%H%M%S")
    end_dt   = datetime.strptime(endtime, "%H%M%S")

    # collect files
    files = sorted(glob.glob(os.path.join(folder, "*.png")))

    time_ranges = []
    for i in range(len(files) - (window_size - 1)):
        # define primary time (middle frame)
        primary_file = files[i + window_size//2]
        primary_time = extract_time_from_filename(primary_file)

        # keep only if primary time is within start/end
        if start_dt <= primary_time <= end_dt:
            range_start = extract_time_from_filename(files[i])
            range_end   = extract_time_from_filename(files[i + window_size - 1])
            time_ranges.append(f"{range_start}-{range_end}")

    return time_ranges


def make_time_list_sliding_window(folder, output_txt):
    """
    Purpose:
        - Creates a list of time ranges from the files in the specified folder using a sliding window
        - Labels the time window by the primary frames from each wavelength
        - Saves the time ranges to a text file
    """
    
    print("Making list of time ranges using sliding window between image captures...")
    
    lambdas = ['0428', '0558', '0630']
    file_lists = []

    # Get the list of files for each wavelength, sorted by time
    for lam in lambdas:
        folder_lam = os.path.join(folder, lam)
        
        # Get all PNG files and sort them by date/time based on the filename pattern
        file_list = sorted(glob.glob(os.path.join(folder_lam, '*_*.png')), 
                           key=lambda x: re.search(r'_(\d{8}_\d{6})', x).group(1) if re.search(r'_(\d{8}_\d{6})', x) else '')
        file_lists.append(file_list)

    grouped_files = []

    # Create sliding window groups (primary + before and after for each wavelength)
    for i in range(1, len(file_lists[0]) - 1):  # Skipping the first and last images
        group = []
        for file_list in file_lists:
            group.extend([file_list[i - 1], file_list[i], file_list[i + 1]])  # previous, primary, next
        grouped_files.append(group)
        
    time_ranges = []

    # Extract time ranges based on primary frames for each wavelength
    for idx, group in enumerate(grouped_files):
        if len(group) == 9:  # 3 frames per wavelength (previous, primary, next for each lambda)
            primary_times = []

            for j in range(1, 9, 3):  # The primary frames are at indices 1, 4, and 7
                match = re.search(r'_(\d{8}_\d{6})', group[j])
                if match:
                    primary_times.append(match.group(1)[8:])  # Extract only time (HHMMSS) portion

            if primary_times:
                # Start time is the earliest of the primary frames, end time is the latest
                start_time = min(primary_times)
                end_time = max(primary_times)
                range_str = f"{start_time}-{end_time}"
                time_ranges.append(range_str)
    
    # Write the time ranges to a text file
    with open(output_txt, 'w') as f:
        for time_range in time_ranges:
            f.write(f"{time_range}\n")

    return time_ranges



def make_time_spreadsheet(output_txt, base_outdir):
    """
    Purpose:
        - creates a spreadsheet that details:
            - when first image from file group was taken
            - when last image from file group was taken
            - total time from first frame to last in co-adding
    """
    
    # Read txt file
    with open(output_txt, 'r') as file:
        lines = file.readlines()
        
    # Initialize to store
    starts = []
    ends = []
    diffs = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        start, end = line.split('-')
        start = start.lstrip('_')
        end = end.lstrip('_')
        
        # Convert from text to integer
        start_int = int(start)
        end_int = int(end)
        
        # Find time of co-added groups (time cadence between frames)
        diff = end_int - start_int
        
        # Append into columns
        starts.append(start_int)
        ends.append(end_int)
        diffs.append(diff)
        
        # Make into pds dataframe
        df = pd.DataFrame({
            'Start': starts,
            'End': ends,
            'Difference': diffs})
        
        output_csv_fn = 'time_ranges.csv'
        output_csv = os.path.join(base_outdir, output_csv_fn)
        
        # Save as csv
        df.to_csv(output_csv, index=False)
    
    return df


def file_data(date, starttime, endtime, maglatsite, folder, output_txt, base_outdir):
    """
    Purpose:
        - runs all filing processes from functions above
    """ 
    
    print("Running all data wrangling processes...")
    
    # Main function calls
    load_local_imagery(folder)
    folder = sort_pngs(folder)
    folder = png_2_h5(folder)
    #folder = group_frames(folder)
    folder = group_by_timestamp(folder, base_outdir)
    #make_time_list(folder, output_txt)
    make_time_list_sliding_window(folder, output_txt)
    #df = make_time_spreadsheet(output_txt, base_outdir)

    lambdas = ['0428/', '0558/', '0630/']
    grouped_files = []

    for lam in lambdas:
        lam_folder = os.path.join(folder, lam, 'h5_files')
        h5_files = glob.glob(os.path.join(lam_folder, '*_*.h5'))
        
        # Sort h5s by timestamp in filenames
        def get_timestamp(file):
            match = re.search(r'_(\d{8}_\d{6})', file)
            return match.group(1) if match else '' # just an empty string if nothing there
        
        h5_list = sorted(h5_files, key=get_timestamp)
        
        # Group into sets of 3
        grouped_h5 = [h5_list[i: i + 3] for i in range(0, len(h5_list), 3)]
        
        for files in grouped_h5:
            if len(files) > 0:
                times = []
                for file in files:
                    match = re.search(r'_(\d{8}_\d{6})', file)
                    if match:
                        time_str = match.group(1)[8:]
                        times.append(time_str)

                if times:
                    min_time = min(times)
                    max_time = max(times)
                    range_str = f"{min_time}-{max_time}"
                    grouped_files.append(os.path.join(lam_folder, f"{range_str}.h5"))

    return date, starttime, endtime, maglatsite, folder, base_outdir, lambdas


def get_grouped_files(folder, lambdas):
    """
    Purpose:
        - pulls together all of the grouped files
    """ 
    
    grouped_files = {lam: sorted(glob.glob(os.path.join(folder, lam, 'h5_files', 'grouped_h5_files', '*.h5')), key=lambda x: int(re.search(r'_(\d+)', os.path.basename(x)).group(1))) for lam in lambdas}
    
    return grouped_files


def process_grouped_files(date, starttime, endtime, maglatsite, folder, base_outdir, lambdas):
    """
    Purpose:
        - pulls info from get_grouped_files and feeds all of the processed h5s into feed_data function
        - this is what allows for the process to be time varying!!
    """ 
    grouped_files = get_grouped_files(folder, lambdas)
    
    # Find the maximum number of groups across all wavelengths
    max_groups = max(len(files) for files in grouped_files.values())
    
    # Iterate over each group index - time varying!!
    for idx in range(max_groups):
        foi_files = {}
        
        for lam in lambdas:
            if idx < len(grouped_files[lam]):
                foi = grouped_files[lam][idx]
                foi_files[lam] = os.path.join(lam, 'h5_files', 'grouped_h5_files', os.path.basename(foi))
            else:
                foi_files[lam] = None  # handle cases where there may be less files for some wavelengths
                
        print("FOI Files:", foi_files)

        # Ensure all necessary files collected before calling feed_data
        if all(foi is not None for foi in foi_files.values()):
            group_number = f"grouped_{idx + 1}"
            group_outdir = os.path.join(base_outdir, group_number)
            os.makedirs(group_outdir, exist_ok=True)
            
            feed_data(date, maglatsite, folder,
                      foi_files['0428/'],
                      foi_files['0558/'],
                      foi_files['0630/'],
                      group_outdir, 
                      group_number)
        else:
            continue
