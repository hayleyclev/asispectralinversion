from filing import file_data
from filing import process_grouped_files

"""
Purpose of this script:
- This is the top-level run script to go through the entire process of:
    - sorting ASI data inputs
    - preprocessing images to denoise them
    - invert imagery + GLOW lookup table to produce maps of
      Q, E0, SigmaP, and SigmaH
    - perform a series of interpolations, smoothing, and
      transformations on these maps

Inputs:
- date
- start time
- end time
- latitude of imager site
- wavelengths of imager filters
- file path to folder containing ASI imagery + GLOW outputs
- file path to folder for storing outputs

Outputs:
- 2D maps (lat, lon) of Q, E0, SigmaP, and SigmaP for all times
- h5 files containing Q, E0, SigmaP, and SigmaH
- CSV describing image grouping + timing cadence
"""

#====USER INPUTS===============================================================

# Date in format YYYYMMDD
date = '20140219'

# Time range in format HHMMSS
# Now supports crossing archive hour boundaries!!
starttime = '132200'
endtime = '132300'

# Magnetic latitude of camera site
maglatsite = 65.8

# Required wavelengths for inversion
# One of each is required per valid image group
lambdas = ['0428', '0558', '0630']

# Folder containing:
    # - downloaded archive imagery
    # - GLOW lookup tables
folder = '/Users/clevenger/Projects/asi_testing/inputs/'

# Output directory for:
    # - inversion outputs
    # - grouped timing CSV
    # - intermediate products
base_outdir = '/Users/clevenger/Projects/asi_testing/outputs/'

# CSV describing:
    # - which files belong to which stack
    # - start/end time of each group
    # - cadence duration including next-cycle return time
output_csv = '/Users/clevenger/Projects/asi_testing/outputs/time_ranges.csv'



#====MAIN EXECUTION - NO USER INPUTS HERE======================================

if __name__ == "__main__":

    print("==========================================")
    print("Starting ASI spectral inversion pipeline!")
    print("==========================================")

    print(f"Date: {date}")
    print(f"Time range: {starttime} -> {endtime}")
    print(f"Input folder: {folder}")
    print(f"Output folder: {base_outdir}")
    print("")


    (
        date,
        starttime,
        endtime,
        maglatsite,
        folder,
        base_outdir,
        lambdas
    ) = file_data(
        date=date,
        starttime=starttime,
        endtime=endtime,
        maglatsite=maglatsite,
        folder=folder,
        output_csv=output_csv,
        base_outdir=base_outdir
    )

    process_grouped_files(
        date=date,
        starttime=starttime,
        endtime=endtime,
        maglatsite=maglatsite,
        folder=folder,
        base_outdir=base_outdir,
        lambdas=lambdas
    )

    print("")
    print("==========================================")
    print("Pipeline complete!")
    print("==========================================")