from filing3 import file_data
from filing3 import png_2_h5
from filing3 import group_frames
from filing3 import process_grouped_files

"""
Purpose of this script:
    - This is the top-level run script to go through the entire process of:
        - sorting ASI data inputs
        - preprocessing images to denoise them
        - invert imagery + GLOW lookup table to produce maps of Q, E0, SigmaP, and SigmaH
        - perform a series of interpolations, smoothing, and transformations on these maps
    - Inputs:
        - date
        - latitude of imager site
        - wavelengths of imager filters
        - file path to folder containing ASI PNGs and GLOW lookup tables
        - file path to folder for storing outputs
        - file path + name for txt file containing information about framerate/time between processed images
    - Outputs:
        - 2D maps (lat, lon) of Q, E0, SigmaP, and SigmaH for all times
        - h5 files containing information about Q, E0, SigmaP, and Sigma H for all times in geodetic and geomagnetic coords
        - txt file containing information about framerate/time between processed images
    """

# Tweakable inputs
date = '20230227'  # date in the form of YYYYMMDD
starttime = '083500'  # time in the format of HHMMSS
endtime = '084000'  # time in the format of HHMMSS

mask_artifact = input("Do you want to mask any artifacts? (y/n): ").lower() == 'y'

maglatsite = 65.8  # site of camera in magnetic latitude

lambdas = ['0428', '0558', '0630']  # wavelengths (nm) of imager filters

folder = '/Users/clevenger/Projects/paper01/sop23_data/202302/27/asi/crossing1/in/'  # folder that holds all image files and GLOW outputs for an hour's worth of an event
base_outdir = '/Users/clevenger/Projects/paper01/sop23_data/202302/27/asi/crossing1/out'  # output directory to store all output figures and h5s
output_txt = '/Users/clevenger/Projects/paper01/sop23_data/202302/27/asi/crossing1/out/time_ranges.txt'  # output for txt file that shows time cadence

# Main function calls to run through entire process
date, starttime, endtime, maglatsite, folder, base_outdir, lambdas = file_data(date, starttime, endtime, maglatsite, folder, output_txt, base_outdir, mask_artifact)

# Convert PNGs to HDF5 after masking and interpolation
png_2_h5(folder)  # Pass lambdas

# Group H5 files
group_frames(folder)  # Pass lambdas

# Process grouped files (inversion, etc.)
process_grouped_files(date, starttime, endtime, maglatsite, folder, base_outdir, lambdas)
