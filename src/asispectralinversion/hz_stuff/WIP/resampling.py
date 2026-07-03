import gemini3d
import gemini3d.model
import os
import numpy as np
from gemini_mapping import gemini_remap
import csv
from asi_to_lompe_filing import rename_files

asi_times_fn = '/Users/clevenger/Projects/data_assimilation2/test_dates/02162023/data_product_outputs/asi_spectral_inversion/10UT/time_ranges.csv'

# read in file (spreadsheet)

# read in directory of asi files
asi_direc = ''

# do all gemini mapping stuff
gemini_remap()

# read in all times
with open(asi_times_fn, newline='') as csv:
    # find column with times
    # store times as df or array?


# change filename of each file from group number format to timerange format
rename_files()

# associate each renamed file with a row in the asi_times_fn spreadsheet

# resample to 20 seconds - will require an interpolation using the data from each
    # start at top of hour
    # 100000
    # 100020
    # 100040
    # ...
    # 100059
    
# 
