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
    - handle input data
        - GLOW lookup table for a specific hour of an event
        - ASI PNGs from Don's archive (http://optics.gi.alaska.edu/amisr_archive/PKR/DASC/PNG/)
    - gets everything into a nice h5 format before preprocessing and inversion
    - contains main function call that feeds processed h5 files into inversion pipeline

Big changes since V3 split:
    - added capability to grab FITS files, not just PNGs
    - had to change around link grabbing to account for both types
    - changed order of artifact removal stuff to now happen first so all smoothed images get saved BEFORE inversion
    - changed so first frame regardless of color is taken to maximize number of stacks
"""

# suppress warnings for archive pulls
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

"""
Timing + Filename Generation
"""

def seconds_since_midnight(timestr):
    """
    Purpose:
    - Turns 24 hour string time into float of seconds past midnight
    """
    print("Finding seconds since midnight...")
    
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


def shift_time(timestr, shift_min):
    """
    Purpose:
        - Shifts a string of time 'hhmmss' by some number of minutes
        - Can use fractional minutes (not intended for that) but should
        - Make a whole number of seconds, then write string to pass along
    """
    
    print("Accounting for time shift...")
    
    hr = float(timestr[:2])
    mn = float(timestr[2:4])
    sc = float(timestr[4:])
    
    t = 60 * 60 * hr + 60 * mn + sc
    t += 60 * shift_min
    
    hr = np.floor( t / (60 * 60) )
    mn = np.floor( (t - 60 * 60 * hr) / 60 )
    sc = np.floor( t - 60 * 60 * hr - 60 * mn )
    
    def num2str(num):
        """
        Purpose:
            - Converts type num to str
        """
        if num >= 10:
            strout = str(int(num))
        else:
            strout = '0' + str(int(num))
        
        return strout
    
    return num2str(hr) + num2str(mn) + num2str(sc)


"""
Downloading
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

        # RAW/FITS (with hour folders)
        f"https://optics.gi.alaska.edu/amisr_archive/PKR/DASC/RAW/{year}/{date}/{hour}",

        # RAW/FITS (without hour folders)
        f"https://optics.gi.alaska.edu/amisr_archive/PKR/DASC/RAW/{year}/{date}",
    ]


# old scheme for grabbing links (keeping for historical purposes - still works for PNGs!)
"""
def genlinks(date, starttime, endtime):
    """ """
    Purpose:
        - Given a date, start and end time, finds links to every DASC frame
    """ """
    
    print("Building link to archive...")
    
    startsecs = seconds_since_midnight(starttime) # seconds from midnight of start time
    endsecs = seconds_since_midnight(endtime) # seconds from midnight of end time
    hr = starttime[:2] # hour in string form
    year = date[:4] # year in string form
    
    # Construct base url
    url = 'http://optics.gi.alaska.edu/amisr_archive/PKR/DASC/PNG/' + year + '/' + date + '/' + hr
    
    response = requests.get(url, verify=False, timeout=30)
    
    # Pull and process file list
    #soup = BeautifulSoup(requests.get(url).text,'html.parser')
    soup = BeautifulSoup(response.text, 'html.parser')
    # First 5 links are not actual events
    rawlinks = soup.find_all('a')[5:]
    # Turning into a numpy array
    links = np.asarray(rawlinks).flatten()
    if len(links)==0:
        print('no imagery found')
        raise Exception('no links')
    
    # Extracting the necessary information
    # Each row is: 
    # [seconds since midnight, color, time]
    
    # Initializing array with zeros
    newlinks = np.zeros([len(links), 3])
    for i in range(len(links)):
        time,color = links[i].split('.')[0].split('_')[2:] # time and color from filename
        newlinks[i, 0] = seconds_since_midnight(time) # seconds past midnight
        newlinks[i, 1] = color
        newlinks[i, 2] = time
    
    # Finding time of each frame, separated by color
    bluesecs = newlinks[:, 0][np.where(newlinks[:, 1] == 428)]
    greensecs = newlinks[:, 0][np.where(newlinks[:, 1] == 558)]
    redsecs = newlinks[:, 0][np.where(newlinks[:, 1] == 630)]
    
    # We arbitrarily choose our first frame to be blue so that the frames are taken in 'b,g,r' order:
    try:
        s0 = bluesecs[np.where(bluesecs < startsecs)[0][-1]]
    except:
        s0 = bluesecs[0]
    # Therefore our last frame must be red
    try:
        s1 = redsecs[np.where(redsecs > endsecs)[0][0]]
    except:
        s1 = redsecs[-1]
        
    # We pull out the indices of the first and last frames we want to download
    startind = np.where(newlinks[:, 0] >= s0)[0][0]
    endind = np.where(newlinks[:, 0] <= s1)[0][-1]
    
    linkstrim = list(links[startind: endind + 1])
    linksout = [url + '/' + link for link in linkstrim]
    return linksout, linkstrim
"""


def get_links_for_hour(date, hour):
    urls = build_hour_urls(date, hour)

    links = []
    seen = set()
    archive_reachable = False

    for url in urls:
        print(f"Checking archive for DASC images: {url}")

        try:
            r = requests.get(url, verify=False, timeout=30)
            archive_reachable = True

            if r.status_code != 200:
                print(f"Archive responded but not reachable ({r.status_code}): {url}")
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
    """
    Purpose:
        - Pulls all DASC png (or FITS!) imagery between <starttime> and <endtime> (UT) on the date of 
          <date>. An example input would be download_imagery('20230314','0730','0745')
    """
    
    print("Downloading imagery from archive...")

    # folder should already exist, but if not:
    os.makedirs(folder, exist_ok=True)

    # Start hour and end hour are the same
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

    #print(f"Files selected: {len(all_links)}")

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
Convert from FITS/PNG to H5
"""

def convert_fits_to_png(folder):

    #print("Checking for FITS files...")

    fits_files = glob.glob(os.path.join(folder, "*.fits"))
    fits_files += glob.glob(os.path.join(folder, "*.FITS"))

    if len(fits_files) == 0:
        #print("No FITS files found.")
        return

    #print(f"Converting {len(fits_files)} FITS files to PNG")

    for fits_file in fits_files:
        png_file = os.path.splitext(fits_file)[0] + ".png"

        if exists(png_file):
            #print(f"PNG already exists: {os.path.basename(png_file)}")
            continue

        try:
            with fits.open(fits_file) as hdul:
                data = hdul[0].data

            if data is None:
                #print(f"No data found in: {os.path.basename(fits_file)}")
                continue

            data = np.array(data, dtype=np.float32)
            data = np.nan_to_num(data)
            data[data < 0] = 0

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

            #print(f"Converted: {os.path.basename(fits_file)}")

        except Exception as e:
            print(f"Failed converting {fits_file}: {e}")


def sort_pngs(folder):
    """
    Moves PNG files into subfolders based on the wavelength in their filenames.
    """
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
    """
    Purpose:
        - reads sorted PNG files
        - creates a new subfolder to store hdf5 files
        - converts each PNG file into an hdf5 file
        - stores converted files into new hdf5 folder
    """
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

    print("Converting FITS files directly to HDF5...")

    fits_files = glob.glob(os.path.join(folder, "*.fits"))
    fits_files += glob.glob(os.path.join(folder, "*.FITS"))

    if len(fits_files) == 0:
        #print("No FITS files found for direct HDF5 conversion.")
        return folder

    for fits_file in fits_files:
        fname = os.path.basename(fits_file)

        wavelength = extract_wavelength_from_filename(fname)
        if wavelength is None:
            #print(f"Could not determine wavelength for FITS file: {fname}")
            continue

        lam_folder = os.path.join(folder, wavelength)
        h5_folder = os.path.join(lam_folder, "h5_files")

        os.makedirs(h5_folder, exist_ok=True)

        h5_name = os.path.splitext(fname)[0] + ".h5"
        h5_path = os.path.join(h5_folder, h5_name)

        if exists(h5_path):
            #print(f"Skipping existing FITS H5: {h5_name}")
            continue

        try:
            with fits.open(fits_file) as hdul:
                data = hdul[0].data

            if data is None:
                #print(f"No data found in FITS file: {fname}")
                continue

            data = np.array(data, dtype=np.float32)
            data = np.nan_to_num(data)

            with h5py.File(h5_path, "w") as h5:
                h5.create_dataset("data", data=data)

            #print(f"Converted FITS to H5: {h5_name}")

        except Exception as e:
            print(f"Failed FITS to h5 conversion for {fname}: {e}")

    return folder


def remove_artifacts_from_single_h5_files(folder, lambdas=("0428", "0558", "0630")):
    """
    Interactively remove artifacts from each single-frame H5 before grouping.

    Saves both:
    - original_data (raw from archive)
    - data (data with artifacts removed)

    The inversion uses:
    - data (data with artifacts removed)
    """

    #print("Starting frame-by-frame artifact removal before grouping...")

    for lam in lambdas:
        h5_folder = os.path.join(folder, lam, "h5_files")

        if not os.path.isdir(h5_folder):
            continue

        h5_files = sorted(glob.glob(os.path.join(h5_folder, "*.h5")))

        for h5_file in h5_files:
            if "grouped_h5_files" in h5_file:
                continue

            with h5py.File(h5_file, "r+") as h5:
                if h5.attrs.get("artifact_removed", False):
                    #print(f"Skipping already-cleaned file: {os.path.basename(h5_file)}")
                    continue

                data = h5["data"][:]

                if "original_data" not in h5:
                    h5.create_dataset("original_data", data=data)

                cleaned = remove_artifacts(data)

                del h5["data"]
                h5.create_dataset("data", data=cleaned.astype(np.float32))

                h5.attrs["artifact_removed"] = True

    #print("Finished artifact removal.")


"""
Grouping after sorting
"""

def build_rgb_groups(folder):
    """
    Build RGB groups in true chronological order:
        - walks through archive order
        - there is no forced starting color (so we are not constrained by losing an image if it doesnt start with a particular wavelength)    
        - group closes once there is one each of 0428, 0558, 0630
        - 0-2 leftovers discarded at the end
    """

    #print("Building RGB stacks from HDF5 files...")

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

    #print(f"Valid RGB groups found: {len(groups)}")

    if len(groups) == 0:
        raise RuntimeError("NO VALID RGB GROUPS FOUND, CANNOT CONTINUE TO INVERSION PIPELINE!")

    return groups


def create_grouped_h5_files(folder, groups):
    """
    Creates the grouped h5 files that get passed to prepare_data.py:

    Each grouped h5 contains:
    - artifact-removed data (3 stacked frames)            
    - original/raw data (3 stacked frames)
    """

    print("Creating grouped h5 files for inversion...")

    for lam in ["0428", "0558", "0630"]:
        grouped_dir = os.path.join(folder, lam, "h5_files", "grouped_h5_files")
        os.makedirs(grouped_dir, exist_ok=True)

    for idx, group in enumerate(groups, start=1):
        for lam in ["0428", "0558", "0630"]:
            single_h5 = os.path.join(folder, lam, "h5_files", group[lam]["filename"])

            grouped_h5 = os.path.join(folder, lam, "h5_files", "grouped_h5_files", f"grouped_{idx}.h5")

            if not exists(single_h5):
                #print(f"Missing h5 file: {single_h5}")
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

    #print("Grouped h5 files created!")


"""
Make spreadsheet to detail timing + how groups are set up
"""

def make_time_spreadsheet(groups, output_csv):
    """
    Create a .csv that shows:
    - files in each RGB stack
    - timing of each RGB stack
    - cadence including return to next stack start
    - includes time from filter switching in between stacks since there is lag going from (i think) B to R
    """

    #print("Creating timing CSV...")

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

    print(f"Saved csv: {output_csv}")

    return df


"""
Pass downloaded/converted/sorted/grouped data into inversion pipeline
"""

def file_data(date, starttime, endtime, maglatsite, folder, output_csv, base_outdir):

    print("Running all data preparation steps!")

    download_imagery(date, starttime, endtime, folder)

    fits_files = glob.glob(os.path.join(folder, "*.fits"))
    fits_files += glob.glob(os.path.join(folder, "*.FITS"))

    png_files = glob.glob(os.path.join(folder, "*.png"))
    png_files += glob.glob(os.path.join(folder, "*.PNG"))

    convert_fits_to_png(folder)
    sort_pngs(folder)

    if len(fits_files) > 0:
        print("Using FITS to HDF5 for inversion (FITS files found in archive)")
        fits_2_h5(folder)
    elif len(png_files) > 0:
        print("Using PNG to HDF5 for inversion (PNG files found in archive)")
        png_2_h5(folder)
    else:
        raise RuntimeError("NO FITS OR PNG FILES AVAILABLE FOR HDF5 CONVERSION")

    remove_artifacts_from_single_h5_files(folder)

    groups = build_rgb_groups(folder)
    create_grouped_h5_files(folder, groups)

    make_time_spreadsheet(groups, output_csv)

    lambdas = ["0428", "0558", "0630"]

    return (date, starttime, endtime, maglatsite, folder, base_outdir, lambdas)


def get_grouped_files(folder, lambdas):
    """
    Return grouped H5 paths to pass into prepare_data.py
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
            rel_path = os.path.relpath(full_path, folder)
            relative_grouped[lam].append(rel_path)

    return relative_grouped


def process_grouped_files(date, starttime, endtime, maglatsite, folder, base_outdir, lambdas, inversion_config=None, skymap_path=None):
    """
    "Main" function for all filing to send off to preparation.py
    """

    grouped_files = get_grouped_files(folder, lambdas)

    max_groups = max(len(files) for files in grouped_files.values())

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

        feed_data(date, maglatsite, folder, foi_files["0428"], foi_files["0558"], foi_files["0630"], 
                  group_outdir, group_number, inversion_config=inversion_config, skymap_path=skymap_path)
        