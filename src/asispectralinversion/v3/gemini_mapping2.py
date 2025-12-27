from gemini3d.grid.convert import geog2geomag
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy
import os
import re
import datetime as dt
import shutil

# LOCATE DIRECTORY CONTAINING FILES FROM ASISPECTRALINVERSION RUN
asi_direc = '/Users/clevenger/Projects/paper01/events/20230227/asi/0835_0840/'
outdir = '/Users/clevenger/Projects/paper01/events/20230227/gemini_inputs/asi/'

# ENTER IN AMOUNT OF TIME COVERED AS A TIME RANGE
start_time = dt.datetime(2023, 2, 27, 8, 35, 9)  # start time
end_time = dt.datetime(2023, 2, 27, 8, 39, 49)     # end time

total_time = end_time - start_time  # total time range

# FIND OUT HOW MANY GROUPED FRAMES THERE WERE FROM THE RUN                                   
fn_convention = r'grouped_\d+geodetic_Q_E0\.h5'
grouped_files = []

for asifn in os.listdir(asi_direc):
    if re.match(fn_convention, asifn):
        grouped_files.append(asifn)

no_groups = len(grouped_files)

# CREATE A VARIABLE THAT IS JUST THE NUMBER OF GROUPS EVENLY SPACED 
# WITHIN THE GIVEN TIME RANGE            
spacing = total_time / (no_groups - 1)  # spacing between each file

# RENAME FILES TO MATCH CONVENTION OF TIME RANGE TO MAKE IT EASY FOR GEMINI TO INGEST
for i, file_name in enumerate(grouped_files):
    # Calculate the new time for the file
    new_time = start_time + (i * spacing)
    
    # Format the new filename
    new_filename = new_time.strftime('%Y%m%d_%H%M%S') + '.h5'
    
    # Define the full old file path
    old_file_path = os.path.join(asi_direc, file_name)
    new_file_path = os.path.join(outdir, new_filename)  # Save to output directory
    
    # Copy the file instead of renaming it
    shutil.copy(old_file_path, new_file_path)

    # Read the copied HDF5 file
    with h5py.File(new_file_path, "r") as h5:  
        lon = h5['Geodetic Longitude'][:]
        lat = h5['Geodetic Latitude'][:]
        Q = h5['Q'][:]
        E0 = h5['E0'][:]
        
        print("Pre-GEMINI Q: ", Q)
        print("Pre-GEMINI E0: ", E0)

    # Converting from geodetic to GEMINI's internal magnetic coordinate system
    print("Converting from geodetic coordinates to GEMINI's internal magnetic coordinate system...")
    phi, theta = geog2geomag(lon, lat)
    gemini_mag_lon = phi * 180 / np.pi
    gemini_mag_lat = 90 - (theta * 180 / np.pi)

    # Grid sampling steps / Creation of target grid/set of locations
    mloni = np.linspace(gemini_mag_lon.min(), gemini_mag_lon.max(), Q.shape[0])
    mlati = np.linspace(gemini_mag_lat.min(), gemini_mag_lat.max(), Q.shape[1])
    MLONi, MLATi = np.meshgrid(mloni, mlati, indexing="ij")

    # Flattening the data for interpolation
    gemini_mag_lat_flat = gemini_mag_lat.flatten()
    gemini_mag_lon_flat = gemini_mag_lon.flatten()
    Q_flat = Q.flatten()
    E0_flat = E0.flatten()

    # Interpolating the data
    Q_gridding = scipy.interpolate.griddata((gemini_mag_lon_flat, gemini_mag_lat_flat), Q_flat, (MLONi, MLATi), fill_value=0)
    E0_gridding = scipy.interpolate.griddata((gemini_mag_lon_flat, gemini_mag_lat_flat), E0_flat, (MLONi, MLATi), fill_value=0)

    ilatmin = np.argmin(abs(mlati - 63.5))
    ilatmax = np.argmin(abs(mlati - 68.5))

    ilonmin=np.argmin(abs(mloni - 250.5))
    ilonmax = np.argmin(abs(mloni - 266.5))

    Q_out = Q_gridding[ilonmin:ilonmax, ilatmin:ilatmax]
    E0_out = E0_gridding[ilonmin:ilonmax, ilatmin:ilatmax]
    lat_out = mlati[ilatmin:ilatmax]
    lon_out = mloni[ilonmin:ilonmax]
    
    print("Q going into GEMINI: ", Q_out)
    print("E0 going into GEMINI: ", E0_out)
    
    # Troubleshooting Plots - Visulaziation of what is about to get fed into GEMINI - Regridded
    plt.title('Map of Q in GEMINI Format')
    plt.pcolormesh(lon_out, lat_out, Q_out.transpose(), cmap='plasma', shading="gouraud")
    plt.colorbar(label = 'mW/m$^2$')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    Q_fn_gemini = f'Q_gemini_final_{new_filename[:-3]}.png'
    Q_out_gemini = os.path.join(outdir, Q_fn_gemini)
    plt.savefig(Q_out_gemini, dpi=300, bbox_inches='tight')
    plt.close()

    plt.title('Map of E0 in GEMINI Format')
    plt.pcolormesh(lon_out, lat_out, E0_out.transpose(), cmap='viridis', shading="gouraud")
    plt.colorbar(label = 'eV')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    E0_fn_gemini = f'E0_gemini_final_{new_filename[:-3]}.png'
    E0_out_gemini = os.path.join(outdir, E0_fn_gemini)
    plt.savefig(E0_out_gemini, dpi=300, bbox_inches='tight')
    plt.close()

    # Final output dataset filename
    new_fn = new_time.strftime('%Y%m%d_%H%M%S') + '.h5'
    
    # Save the final dataset to HDF5
    with h5py.File(os.path.join(outdir, new_fn), "w") as hdf:
        hdf.create_dataset("MLAT", data=lat_out, dtype='f')
        hdf.create_dataset("MLON", data=lon_out, dtype='f')
        hdf.create_dataset("Q", data=Q_out, dtype='f')
        hdf.create_dataset("E0", data=E0_out, dtype='f')
        
    print("Q in H5 file: ", Q_out)
    print("E0 in H5 file: ", E0_out)
