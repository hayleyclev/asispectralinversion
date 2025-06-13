#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:19:38 2024

@author: clevenger
"""

import re
import csv
import requests
from bs4 import BeautifulSoup

# URL of the archive page where the files are located
url = 'http://optics.gi.alaska.edu/amisr_archive/PKR/DASC/PNG/2023/20230210/10/'

# Output CSV file
output_csv = '/Users/clevenger/Projects/data_assimilation2/test_dates/02102023/data_product_outputs/asispectralinversion/output.csv'

# Send a request to get the page content
response = requests.get(url)
response.raise_for_status()  # Check for successful request

# Parse the page content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all the links to image files (assuming they have .png extension)
image_links = soup.find_all('a', href=re.compile(r'\.png$'))

# Prepare a list to hold the extracted time and wavelength
time_wavelength_data = []

# Iterate through each image link and extract the time and wavelength
for link in image_links:
    filename = link['href']  # Get the file name from the href attribute
    match = re.search(r'_(\d{8}_\d{6})_(\d{4})\.png', filename)
    if match:
        time = match.group(1)       # Extract timestamp (YYYYMMDD_HHMMSS)
        wavelength = match.group(2) # Extract wavelength (e.g., 0558)
        
        # Append to list in the format [time, wavelength]
        time_wavelength_data.append([time, wavelength])

# Write the extracted data to a CSV file
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['time', 'wavelength'])  # Write header
    writer.writerows(time_wavelength_data)   # Write the extracted rows

print(f"CSV file has been created: {output_csv}")
