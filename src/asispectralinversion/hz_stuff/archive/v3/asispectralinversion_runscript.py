from filing import file_data
from filing import process_grouped_files

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
date = '20230227'
starttime = '083500'
endtime = '083700'

maglatsite = 65.8
lambdas = ['0428/', '0558/', '0630/']

folder = '/Users/clevenger/Projects/asi_testing/inputs/'
base_outdir = '/Users/clevenger/Projects/asi_testing/outputs/'
output_txt = '/Users/clevenger/Projects/asi_testing/outputs/time_ranges.txt'

# Main function calls to run through entire process
date, starttime, endtime, maglatsite, folder, base_outdir, lambdas = file_data(date, starttime, endtime, maglatsite, folder, output_txt, base_outdir)
process_grouped_files(date, starttime, endtime, maglatsite, folder, base_outdir, lambdas)
