from pathlib import Path
import re

import h5py
import numpy as np
import scipy.interpolate
from gemini3d.grid.convert import geog2geomag


def transform_asi_outputs_to_gemini(
    asi_outdir, # LOCATE DIRECTORY CONTAINING FILES FROM ASISPECTRALINVERSION RUN
    gemini_outdir,
    start_datetime, # ENTER IN AMOUNT OF TIME COVERED AS A TIME RANGE
    end_datetime,
    make_plots=False,
    mlat_min=63.5,
    mlat_max=68.5,
    mlon_min=250.5,
    mlon_max=266.5,
):
    asi_outdir = Path(asi_outdir).expanduser().resolve()
    gemini_outdir = Path(gemini_outdir).expanduser().resolve()
    gemini_outdir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"grouped_\d+geodetic_Q_E0\.h5")

    # FIND OUT HOW MANY GROUPED FRAMES THERE WERE FROM THE RUN  
    grouped_files = sorted(
        p for p in asi_outdir.iterdir()
        if p.is_file() and pattern.match(p.name)
    )

    if not grouped_files:
        raise RuntimeError(f"No grouped geodetic Q/E0 files found in:\n{asi_outdir}")

    if len(grouped_files) == 1:
        times = [start_datetime]
    else:
        # CREATE A VARIABLE THAT IS JUST THE NUMBER OF GROUPS EVENLY SPACED 
        # WITHIN THE GIVEN TIME RANGE            
        spacing = (end_datetime - start_datetime) / (len(grouped_files) - 1)
        times = [start_datetime + i * spacing for i in range(len(grouped_files))]

    made_files = []

    for file_path, file_time in zip(grouped_files, times):
        timestamp = file_time.strftime("%Y%m%d_%H%M%S")
        output_path = gemini_outdir / f"{timestamp}.h5"

        # Read the copied HDF5 file
        with h5py.File(file_path, "r") as h5:
            lon = h5["Geodetic Longitude"][:]
            lat = h5["Geodetic Latitude"][:]
            Q = h5["Q"][:]
            E0 = h5["E0"][:]

        # Converting from geodetic to GEMINI's internal magnetic coordinate system
        print("Converting from geodetic coordinates to GEMINI's internal magnetic coordinate system...")
        phi, theta = geog2geomag(lon, lat)
        gem_lon = phi * 180.0 / np.pi
        gem_lat = 90.0 - theta * 180.0 / np.pi

        # Grid sampling steps / Creation of target grid/set of locations
        mloni = np.linspace(gem_lon.min(), gem_lon.max(), Q.shape[0])
        mlati = np.linspace(gem_lat.min(), gem_lat.max(), Q.shape[1])
        MLONi, MLATi = np.meshgrid(mloni, mlati, indexing="ij")

        # Flattening the data for interpolation
        Q_grid = scipy.interpolate.griddata(
            (gem_lon.flatten(), gem_lat.flatten()),
            Q.flatten(),
            (MLONi, MLATi),
            fill_value=0,
        )

        # Interpolating the data
        E0_grid = scipy.interpolate.griddata(
            (gem_lon.flatten(), gem_lat.flatten()),
            E0.flatten(),
            (MLONi, MLATi),
            fill_value=0,
        )

        ilatmin = np.argmin(abs(mlati - mlat_min))
        ilatmax = np.argmin(abs(mlati - mlat_max))
        ilonmin = np.argmin(abs(mloni - mlon_min))
        ilonmax = np.argmin(abs(mloni - mlon_max))

        if ilatmin > ilatmax:
            ilatmin, ilatmax = ilatmax, ilatmin
        if ilonmin > ilonmax:
            ilonmin, ilonmax = ilonmax, ilonmin

        Q_out = Q_grid[ilonmin:ilonmax, ilatmin:ilatmax]
        E0_out = E0_grid[ilonmin:ilonmax, ilatmin:ilatmax]
        lat_out = mlati[ilatmin:ilatmax]
        lon_out = mloni[ilonmin:ilonmax]
        
        print("Q going into GEMINI: ", Q_out)
        print("E0 going into GEMINI: ", E0_out)

        # Save the final dataset to HDF5
        with h5py.File(output_path, "w") as hdf:
            hdf.create_dataset("MLAT", data=lat_out, dtype="f")
            hdf.create_dataset("MLON", data=lon_out, dtype="f")
            hdf.create_dataset("Q", data=Q_out, dtype="f")
            hdf.create_dataset("E0", data=E0_out, dtype="f")
            
        print("Q in H5 file: ", Q_out)
        print("E0 in H5 file: ", E0_out)

        if make_plots:
            make_gemini_plots(lon_out, lat_out, Q_out, E0_out, gemini_outdir, timestamp)

        made_files.append(output_path)

    return made_files

# Troubleshooting Plots - Visulaziation of what is about to get fed into GEMINI - Regridded
def make_gemini_plots(lon_out, lat_out, Q_out, E0_out, outdir, timestamp):
    import matplotlib.pyplot as plt

    outdir = Path(outdir)

    plt.figure()
    plt.title("Map of Q in GEMINI Format")
    plt.pcolormesh(lon_out, lat_out, Q_out.T, cmap="plasma", shading="gouraud")
    plt.colorbar(label="mW/m$^2$")
    plt.xlabel("Geomagnetic Longitude")
    plt.ylabel("Geomagnetic Latitude")
    plt.savefig(outdir / f"Q_gemini_final_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.title("Map of E0 in GEMINI Format")
    plt.pcolormesh(lon_out, lat_out, E0_out.T, cmap="viridis", shading="gouraud")
    plt.colorbar(label="eV")
    plt.xlabel("Geomagnetic Longitude")
    plt.ylabel("Geomagnetic Latitude")
    plt.savefig(outdir / f"E0_gemini_final_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()
