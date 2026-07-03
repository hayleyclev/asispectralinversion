#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:20:08 2026

@author: clevenger
"""

import os
import re
import glob
import shutil
import requests
import urllib3
import numpy as np
import pandas as pd
import h5py
from PIL import Image
from bs4 import BeautifulSoup
from astropy.io import fits
from os.path import exists
from transformation import feed_data
from artifact_removing import remove_artifacts

"""
Purpose of this script:
- Handle ASI input data
- Download imagery from archive
- Support PNG + FITS archive files
- Convert FITS -> PNG when needed
- Convert PNG -> HDF5
- Build correct RGB image groups for inversion
- Create grouped H5 files with frame_1/frame_2/frame_3
- Create timing CSV describing image cadence
- Feed grouped files into inversion pipeline
"""

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

"""
==============================
Timing + filename helpers
==============================
"""

def seconds_since_midnight(timestr):
    return int(timestr[:2]) * 3600 + int(timestr[2:4]) * 60 + int(timestr[4:6])


def extract_timestamp_from_filename(filename):
    # PNG: PKR_YYYYMMDD_HHMMSS_XXXX.png
    m = re.search(r'_(\d{8})_(\d{6})_(\d{4})', filename)
    if m:
        return m.group(2)

    # FITS: PKR_DASC_XXXX_YYYYMMDD_HHMMSS.xxx.FITS
    m = re.search(r'_(\d{8})_(\d{6})\.', filename)
    if m:
        return m.group(2)

    return None


def extract_wavelength_from_filename(filename):
    # PNG
    m = re.search(r'_(\d{4})\.png$', filename, re.IGNORECASE)
    if m:
        return m.group(1)

    # FITS
    m = re.search(r'DASC_(\d{4})_', filename, re.IGNORECASE)
    if m:
        return m.group(1)

    return None


"""
==============================
Download helpers
==============================
"""

def download_no_wget(url, outpath):
    r = requests.get(url, stream=True, verify=False, timeout=180)
    r.raise_for_status()

    with open(outpath, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)


def build_hour_urls(date, hour):
    year = date[:4]

    return [
        # PNG (has hour folders)
        f"https://optics.gi.alaska.edu/amisr_archive/PKR/DASC/PNG/{year}/{date}/{hour}",

        # RAW (sometimes has hour folders)
        f"https://optics.gi.alaska.edu/amisr_archive/PKR/DASC/RAW/{year}/{date}/{hour}",

        # RAW (often no hour folder)
        f"https://optics.gi.alaska.edu/amisr_archive/PKR/DASC/RAW/{year}/{date}",
    ]


def get_links_for_hour(date, hour):
    urls = build_hour_urls(date, hour)

    links = []
    seen = set()
    archive_reachable = False

    for url in urls:
        print(f"Checking archive: {url}")

        try:
            r = requests.get(url, verify=False, timeout=30)
            archive_reachable = True

            if r.status_code != 200:
                print(f"Archive responded but not usable ({r.status_code}): {url}")
                continue

        except requests.exceptions.ConnectionError:
            print("ARCHIVE CANNOT BE REACHED! SERVER MAY BE DOWN")
            continue

        except Exception as e:
            print(f"Error accessing {url}: {e}")
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        raw_links = soup.find_all("a")

        for item in raw_links:
            href = item.get("href")

            if not href:
                continue

            if href.startswith("?") or href.startswith("/"):
                continue

            if not (href.lower().endswith(".png") or href.lower().endswith(".fits")):
                continue

            if href in seen:
                continue

            seen.add(href)
            links.append((f"{url}/{href}", href))

    if not archive_reachable:
        print("ARCHIVE CANNOT BE REACHED! SERVER MAY BE DOWN")

    return links


def download_imagery(date, starttime, endtime, folder):
    print("Downloading imagery from archive...")

    os.makedirs(folder, exist_ok=True)

    start_hour = int(starttime[:2])
    end_hour = int(endtime[:2])

    all_links = []

    for hour in range(start_hour, end_hour + 1):
        hour_str = f"{hour:02d}"
        hour_links = get_links_for_hour(date, hour_str)

        for full_url, fname in hour_links:
            ts = extract_timestamp_from_filename(fname)
            if ts is None:
                continue

            if hour == start_hour and ts < starttime:
                continue
            if hour == end_hour and ts > endtime:
                continue

            all_links.append((full_url, fname))

    print(f"Files selected: {len(all_links)}")

    if len(all_links) == 0:
        raise RuntimeError(
            "NO IMAGES FOUND, CANNOT CONTINUE TO INVERSION PIPELINE"
        )

    for url, fname in all_links:
        outpath = os.path.join(folder, fname)

        if exists(outpath):
            print(f"Skipping existing file: {fname}")
            continue

        print(f"Downloading: {fname}")
        download_no_wget(url, outpath)

    return folder


"""
==============================
Conversion + sorting
==============================
"""

def convert_fits_to_png(folder):
    """
    Convert FITS files to PNG using a percentile contrast stretch
    so raw FITS images look less grainy than simple min/max scaling.
    """

    print("Checking for FITS files...")

    fits_files = glob.glob(os.path.join(folder, "*.fits"))
    fits_files += glob.glob(os.path.join(folder, "*.FITS"))

    if len(fits_files) == 0:
        print("No FITS files found.")
        return

    print(f"Converting {len(fits_files)} FITS files to PNG")

    for fits_file in fits_files:
        png_file = os.path.splitext(fits_file)[0] + ".png"

        if exists(png_file):
            print(f"PNG already exists: {os.path.basename(png_file)}")
            continue

        try:
            with fits.open(fits_file) as hdul:
                data = hdul[0].data

            if data is None:
                print(f"No data found in: {os.path.basename(fits_file)}")
                continue

            # Convert FITS data to native float before math.
            # This avoids dtype errors from big-endian integer FITS arrays.
            data = np.array(data, dtype=np.float32)
            data = np.nan_to_num(data)

            # Remove obvious negative/background offset
            data[data < 0] = 0

            # Percentile stretch handles hot pixels/outliers better than min/max
            low, high = np.percentile(data, (0.5, 99.8))

            if high > low:
                data = np.clip(data, low, high)
                data = (data - low) / (high - low)
            else:
                data = data - np.min(data)
                max_val = np.max(data)

                if max_val > 0:
                    data = data / max_val

            data = (255 * data).astype(np.uint8)

            img = Image.fromarray(data)
            img.save(png_file)

            print(f"Converted: {os.path.basename(fits_file)}")

        except Exception as e:
            print(f"Failed converting {fits_file}: {e}")


def sort_pngs(folder):
    for f in os.listdir(folder):
        if not f.lower().endswith(".png"):
            continue

        lam = extract_wavelength_from_filename(f)
        if lam is None:
            continue

        sub = os.path.join(folder, lam)
        os.makedirs(sub, exist_ok=True)

        shutil.move(os.path.join(folder, f), os.path.join(sub, f))

    return folder


def png_2_h5(folder):
    for lam in ["0428", "0558", "0630"]:
        lam_dir = os.path.join(folder, lam)
        if not os.path.isdir(lam_dir):
            continue

        h5_dir = os.path.join(lam_dir, "h5_files")
        os.makedirs(h5_dir, exist_ok=True)

        for f in os.listdir(lam_dir):
            if not f.endswith(".png"):
                continue

            png = os.path.join(lam_dir, f)
            h5 = os.path.join(h5_dir, f.replace(".png", ".h5"))

            if exists(h5):
                continue

            arr = np.array(Image.open(png))
            with h5py.File(h5, "w") as hf:
                hf.create_dataset("data", data=arr)

    return folder

def fits_2_h5(folder):
    """
    Convert FITS files directly to HDF5 for science/inversion use.

    This preserves the FITS intensity data much better than:
        FITS -> PNG -> HDF5

    Output:
        folder/<wavelength>/h5_files/<original_fits_name>.h5
    """

    print("Converting FITS files directly to HDF5...")

    fits_files = glob.glob(os.path.join(folder, "*.fits"))
    fits_files += glob.glob(os.path.join(folder, "*.FITS"))

    if len(fits_files) == 0:
        print("No FITS files found for direct HDF5 conversion.")
        return folder

    for fits_file in fits_files:
        fname = os.path.basename(fits_file)

        wavelength = extract_wavelength_from_filename(fname)
        if wavelength is None:
            print(f"Could not determine wavelength for FITS file: {fname}")
            continue

        lam_folder = os.path.join(folder, wavelength)
        h5_folder = os.path.join(lam_folder, "h5_files")

        os.makedirs(h5_folder, exist_ok=True)

        h5_name = os.path.splitext(fname)[0] + ".h5"
        h5_path = os.path.join(h5_folder, h5_name)

        if exists(h5_path):
            print(f"Skipping existing FITS H5: {h5_name}")
            continue

        try:
            with fits.open(fits_file) as hdul:
                data = hdul[0].data

            if data is None:
                print(f"No data found in FITS file: {fname}")
                continue

            # Preserve science data as float32.
            # Do NOT percentile stretch, gamma correct, or convert to uint8.
            data = np.array(data, dtype=np.float32)
            data = np.nan_to_num(data)

            with h5py.File(h5_path, "w") as h5:
                h5.create_dataset("data", data=data)

            print(f"Converted FITS -> H5: {h5_name}")

        except Exception as e:
            print(f"Failed FITS -> H5 for {fname}: {e}")

    return folder


def remove_artifacts_from_single_h5_files(folder, lambdas=("0428", "0558", "0630")):
    """
    Interactively remove artifacts from each single-frame H5 before grouping.

    Saves both:
    - original_data
    - data

    The inversion uses:
    - data
    """

    print("Starting per-frame artifact removal before grouping...")

    for lam in lambdas:
        h5_folder = os.path.join(folder, lam, "h5_files")

        if not os.path.isdir(h5_folder):
            continue

        h5_files = sorted(glob.glob(os.path.join(h5_folder, "*.h5")))

        for h5_file in h5_files:
            if "grouped_h5_files" in h5_file:
                continue

            print("")
            print(f"Reviewing artifact removal for: {os.path.basename(h5_file)}")

            with h5py.File(h5_file, "r+") as h5:
                if h5.attrs.get("artifact_removed", False):
                    print(f"Skipping already-cleaned file: {os.path.basename(h5_file)}")
                    continue

                data = h5["data"][:]

                if "original_data" not in h5:
                    h5.create_dataset("original_data", data=data)

                cleaned = remove_artifacts(data)

                del h5["data"]
                h5.create_dataset("data", data=cleaned.astype(np.float32))

                h5.attrs["artifact_removed"] = True

            print(f"Saved original + cleaned frame: {os.path.basename(h5_file)}")

    print("Finished per-frame artifact removal.")


"""
Helper functions for grouping, post-sorting
"""

def build_rgb_groups(folder):
    """
    Build groups in true chronological order

    How it happens:
    - walks through archive order
    - there is no forced starting color (so we are not constrained by losing an image if it doesnt start with a particular wl)    
    - group closes once there is one each of 0428, 0558, 0630
    - 0-2 leftovers discarded at the end
    """

    print("Building sequential RGB groups from HDF5 files...")

    all_files = []

    for lam in ["0428", "0558", "0630"]:
        h5_folder = os.path.join(folder, lam, "h5_files")

        if not os.path.isdir(h5_folder):
            continue

        for file in os.listdir(h5_folder):
            if not file.lower().endswith(".h5"):
                continue

            # Do not group already-grouped outputs
            if file.startswith("grouped_"):
                continue

            timestamp = extract_timestamp_from_filename(file)

            if timestamp is None:
                continue

            all_files.append({
                "filename": file,
                "timestamp": timestamp,
                "seconds": seconds_since_midnight(timestamp),
                "wavelength": lam
            })

    all_files = sorted(all_files, key=lambda x: x["seconds"])

    groups = []
    current = {}

    for item in all_files:
        lam = item["wavelength"]

        if lam not in current:
            current[lam] = item

        if len(current) == 3:
            groups.append({
                "0428": current["0428"],
                "0558": current["0558"],
                "0630": current["0630"]
            })
            current = {}

    print(f"Valid RGB groups found: {len(groups)}")

    if len(groups) == 0:
        raise RuntimeError(
            "NO VALID RGB GROUPS FOUND, CANNOT CONTINUE TO INVERSION PIPELINE"
        )

    return groups


def create_grouped_h5_files(folder, groups):
    """
    Create grouped H5 files expected by prepare_data.py.

    Each grouped H5 contains:
    - frame_1, frame_2, frame_3              cleaned data
    - original_frame_1, original_frame_2,
      original_frame_3                       original unedited data
    """

    print("Creating grouped H5 files for inversion compatibility...")

    for lam in ["0428", "0558", "0630"]:
        grouped_dir = os.path.join(
            folder,
            lam,
            "h5_files",
            "grouped_h5_files"
        )
        os.makedirs(grouped_dir, exist_ok=True)

    for idx, group in enumerate(groups, start=1):
        for lam in ["0428", "0558", "0630"]:
            single_h5 = os.path.join(
                folder,
                lam,
                "h5_files",
                group[lam]["filename"]
            )

            grouped_h5 = os.path.join(
                folder,
                lam,
                "h5_files",
                "grouped_h5_files",
                f"grouped_{idx}.h5"
            )

            if not exists(single_h5):
                print(f"Missing single H5 file: {single_h5}")
                continue

            with h5py.File(single_h5, "r") as src:
                cleaned_data = src["data"][:]

                if "original_data" in src:
                    original_data = src["original_data"][:]
                else:
                    original_data = cleaned_data.copy()

            with h5py.File(grouped_h5, "w") as dst:
                dst.create_dataset("frame_1", data=cleaned_data)
                dst.create_dataset("frame_2", data=cleaned_data)
                dst.create_dataset("frame_3", data=cleaned_data)

                dst.create_dataset("original_frame_1", data=original_data)
                dst.create_dataset("original_frame_2", data=original_data)
                dst.create_dataset("original_frame_3", data=original_data)

    print("Grouped H5 files created successfully.")


"""
Helper function to make spreadsheet of timing/grouping info
"""

def make_time_spreadsheet(groups, output_csv):
    """
    Create a .csv that shows:
    - files in each RGB stack
    - timing span
    - cadence including return to next stack start
    """

    print("Creating timing CSV...")

    rows = []

    for i, group in enumerate(groups):
        timestamps = sorted([
            group["0428"]["timestamp"],
            group["0558"]["timestamp"],
            group["0630"]["timestamp"]
        ])

        start_time = timestamps[0]
        last_image_time = timestamps[-1]

        if i < len(groups) - 1:
            next_group_start = min([
                groups[i + 1]["0428"]["timestamp"],
                groups[i + 1]["0558"]["timestamp"],
                groups[i + 1]["0630"]["timestamp"]
            ])
        else:
            next_group_start = last_image_time

        duration = (seconds_since_midnight(next_group_start) - seconds_since_midnight(start_time))

        rows.append({
            "group_number": i + 1,
            "blue_0428": group["0428"]["filename"],
            "green_0558": group["0558"]["filename"],
            "red_0630": group["0630"]["filename"],
            "start_time": start_time,
            "last_image_time": last_image_time,
            "next_group_start": next_group_start,
            "duration_seconds": duration
        })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved timing CSV: {output_csv}")

    return df


"""
Helper functions to get downloaded/converted/sorted/grouped data into inversion pipeline
"""

def file_data(date, starttime, endtime, maglatsite, folder, output_csv, base_outdir):
    """
    Run full file preparation pipeline.

    Artifact removal now happens BEFORE grouping/inversion.
    """

    print("Running all data preparation steps...")

    download_imagery(date, starttime, endtime, folder)

    fits_files = glob.glob(os.path.join(folder, "*.fits"))
    fits_files += glob.glob(os.path.join(folder, "*.FITS"))

    png_files = glob.glob(os.path.join(folder, "*.png"))
    png_files += glob.glob(os.path.join(folder, "*.PNG"))

    # Optional visualization branch for FITS-era data
    convert_fits_to_png(folder)
    sort_pngs(folder)

    if len(fits_files) > 0:
        print("Using FITS → HDF5 for inversion")
        fits_2_h5(folder)
    elif len(png_files) > 0:
        print("Using PNG → HDF5 for inversion")
        png_2_h5(folder)
    else:
        raise RuntimeError(
            "NO FITS OR PNG FILES AVAILABLE FOR HDF5 CONVERSION"
        )

    # NEW: clean each frame interactively before grouping
    remove_artifacts_from_single_h5_files(folder)

    groups = build_rgb_groups(folder)
    create_grouped_h5_files(folder, groups)
    make_time_spreadsheet(groups, output_csv)

    lambdas = ["0428", "0558", "0630"]

    return (date, starttime, endtime, maglatsite, folder, base_outdir, lambdas)


def get_grouped_files(folder, lambdas):
    """
    Return grouped H5 paths expected by prepare_data.py
    """

    grouped_files = {
        lam: sorted(glob.glob(os.path.join(folder, lam, "h5_files", "grouped_h5_files", "*.h5")),
        key=lambda x: int(re.search(r'grouped_(\d+)', os.path.basename(x)).group(1)))
        for lam in lambdas
    }

    relative_grouped = {}

    for lam in lambdas:
        relative_grouped[lam] = []

        for full_path in grouped_files[lam]:
            rel_path = os.path.relpath(
                full_path,
                folder
            )
            relative_grouped[lam].append(rel_path)

    return relative_grouped


def process_grouped_files(date, starttime, endtime, maglatsite, folder, base_outdir, lambdas):
    """
    Feed grouped files into inversion pipeline
    """

    print("Processing grouped files for inversion...")

    grouped_files = get_grouped_files(folder, lambdas)

    max_groups = max(
        len(files)
        for files in grouped_files.values()
    )

    for idx in range(max_groups):
        foi_files = {}

        for lam in lambdas:
            if idx < len(grouped_files[lam]):
                foi_files[lam] = grouped_files[lam][idx]
            else:
                foi_files[lam] = None

        if not all(foi_files.values()):
            continue

        group_number = f"grouped_{idx + 1}"

        group_outdir = os.path.join(
            base_outdir,
            group_number
        )
        os.makedirs(group_outdir, exist_ok=True)

        print(f"Running inversion for {group_number}")

        feed_data(
            date,
            maglatsite,
            folder,
            foi_files["0428"],
            foi_files["0558"],
            foi_files["0630"],
            group_outdir,
            group_number
        )