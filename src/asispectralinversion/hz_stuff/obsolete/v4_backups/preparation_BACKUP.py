#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:11:32 2026

@author: clevenger
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import datetime
import os
from apexpy import Apex
from inversion import load_lookup_tables_directory
from inversion import calculate_E0_Q_v2
from preprocessing import wavelet_denoise_resample
from preprocessing import gaussian_denoise_resample
from preprocessing import to_rayleighs
from inversion import calculate_Sig
from artifact_removing import remove_artifacts

"""
Purpose of this script:
    - takes in ASI/GLOW information, preparing for preprocessing and inversion
    - runs preprocessing and inversion functions
    - returns Q, E0, SigmaP, and SigmaH in regularized, geomagnetic coordinates
"""

def copy_h5(vtest):
    """
    Purpose: 
        - copies an HDF5 structure to a python dict recursively
    """
    
    dicttest = {}
    keyslist = list(vtest.keys())
    for key in keyslist:
        if type(vtest[key]) == h5py._hl.dataset.Dataset:
            if vtest[key].shape[1] == 1:
                if vtest[key].shape[0] == 1:
                    dicttest[key] = vtest[key][0][0]
                else:
                    dicttest[key] = np.asarray(vtest[key]).flatten()
            else:
                dicttest[key] = np.asarray(vtest[key])
        else:
            dicttest[key] = copy_h5(vtest[key])
            
    return dicttest


def prepare_data(date, maglatsite, folder, foi_0428, foi_0558, foi_0630, group_outdir, group_number):
    """
    Purpose:
        - prepares Q, E0, SigP, and SigH given ASI data

    Notes:
        - artifact removal is assumed to have already happened in filing.py
        - frame_1/frame_2/frame_3 are cleaned frames used for inversion
        - original_frame_1/original_frame_2/original_frame_3 are used only
          for before/after diagnostic plots
    """

    print("Pulling information from data files and lookup tables...")

    dtdate = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:]))
    blur_deg_EW = 0.4
    blur_deg_NS = 0.04
    n_shifts = 50
    background_method = 'patches'

    v = load_lookup_tables_directory(folder, maglatsite)

    redims = copy_h5(h5py.File(folder + '/' + foi_0630))
    greenims = copy_h5(h5py.File(folder + '/' + foi_0558))
    blueims = copy_h5(h5py.File(folder + '/' + foi_0428))

    vskymap = copy_h5(h5py.File(folder + 'skymap.mat')['magnetic_footpointing'])
    skymapred = [vskymap['180km']['lat'], vskymap['180km']['lon']]
    skymapgreen = [vskymap['110km']['lat'], vskymap['110km']['lon']]
    skymapblue = [vskymap['107km']['lat'], vskymap['107km']['lon']]

    # Cleaned/coadded images used for inversion
    greenimcoadd = (greenims['frame_1'] + greenims['frame_2'] + greenims['frame_3']) / 3
    blueimcoadd = (blueims['frame_1'] + blueims['frame_2'] + blueims['frame_3']) / 3
    redimcoadd = (redims['frame_1'] + redims['frame_2'] + redims['frame_3']) / 3

    # Original/coadded images used only for before/after diagnostic plots
    greenimcoadd_original = (
        greenims.get('original_frame_1', greenims['frame_1']) +
        greenims.get('original_frame_2', greenims['frame_2']) +
        greenims.get('original_frame_3', greenims['frame_3'])
    ) / 3

    blueimcoadd_original = (
        blueims.get('original_frame_1', blueims['frame_1']) +
        blueims.get('original_frame_2', blueims['frame_2']) +
        blueims.get('original_frame_3', blueims['frame_3'])
    ) / 3

    redimcoadd_original = (
        redims.get('original_frame_1', redims['frame_1']) +
        redims.get('original_frame_2', redims['frame_2']) +
        redims.get('original_frame_3', redims['frame_3'])
    ) / 3

    # Save original and cleaned diagnostic images
    plt.imshow(redimcoadd_original)
    plt.title('Red Imagery Original')
    plt.xlabel('E-W')
    plt.ylabel('N-S')
    plt.savefig(os.path.join(group_outdir, 'red_imagery_original.png'))
    plt.close()

    plt.imshow(redimcoadd)
    plt.title('Red Imagery Artifact Removed')
    plt.xlabel('E-W')
    plt.ylabel('N-S')
    plt.savefig(os.path.join(group_outdir, 'red_imagery_artifact_removed.png'))
    plt.close()

    plt.imshow(greenimcoadd_original)
    plt.title('Green Imagery Original')
    plt.xlabel('E-W')
    plt.ylabel('N-S')
    plt.savefig(os.path.join(group_outdir, 'green_imagery_original.png'))
    plt.close()

    plt.imshow(greenimcoadd)
    plt.title('Green Imagery Artifact Removed')
    plt.xlabel('E-W')
    plt.ylabel('N-S')
    plt.savefig(os.path.join(group_outdir, 'green_imagery_artifact_removed.png'))
    plt.close()

    plt.imshow(blueimcoadd_original)
    plt.title('Blue Imagery Original')
    plt.xlabel('E-W')
    plt.ylabel('N-S')
    plt.savefig(os.path.join(group_outdir, 'blue_imagery_original.png'))
    plt.close()

    plt.imshow(blueimcoadd)
    plt.title('Blue Imagery Artifact Removed')
    plt.xlabel('E-W')
    plt.ylabel('N-S')
    plt.savefig(os.path.join(group_outdir, 'blue_imagery_artifact_removed.png'))
    plt.close()

    # Inversion now uses already-cleaned data
    redimcoadd_interp = redimcoadd
    greenimcoadd_interp = greenimcoadd
    blueimcoadd_interp = blueimcoadd

    A = Apex(date=dtdate)
    bmla, bmlo = A.convert(
        skymapblue[0].reshape(-1),
        np.mod(skymapblue[1].reshape(-1), 360),
        'geo',
        'apex',
        height=110
    )

    minmlat = np.amin(bmla[np.where(~np.isnan(bmla))])
    maxmlat = np.amax(bmla[np.where(~np.isnan(bmla))])

    minmlon = np.amin(bmlo[np.where(~np.isnan(bmlo))])
    maxmlon = np.amax(bmlo[np.where(~np.isnan(bmlo))])

    interplonvec = np.linspace(minmlon, maxmlon, 1024)
    interplatvec = np.linspace(minmlat, maxmlat, 1024)

    print("Denoising images...")

    blueimdenoisewavelet, blueimreg, lon110, lat110, maglon, maglat, bluebgbright, bluesig = wavelet_denoise_resample(
        blueimcoadd_interp,
        dtdate,
        skymapblue[1],
        skymapblue[0],
        interplonvec,
        interplatvec,
        110,
        nshifts=n_shifts,
        background_method=background_method,
        plot=False
    )

    blueimdenoisegauss, _, _, _, _, _, _, _ = gaussian_denoise_resample(
        blueimcoadd_interp,
        dtdate,
        skymapblue[1],
        skymapblue[0],
        interplonvec,
        interplatvec,
        110,
        blur_deg_EW,
        NS_deg=blur_deg_NS,
        plot=False
    )

    noise = np.copy(skymapblue[0]).reshape(-1)
    noiseadd = (np.random.randn(len(noise[np.where(~np.isnan(noise))])) * bluesig)
    noise[np.where(~np.isnan(noise))] = noiseadd
    noise = noise.reshape(skymapblue[0].shape)

    redimdenoise, redimreg, _, _, _, _, redbgbright, redsig = wavelet_denoise_resample(
        redimcoadd_interp,
        dtdate,
        skymapred[1],
        skymapred[0],
        interplonvec,
        interplatvec,
        110,
        nshifts=n_shifts,
        background_method=background_method,
        plot=False
    )

    greenimdenoise, greenimreg, _, _, _, _, greenbgbright, greensig = wavelet_denoise_resample(
        greenimcoadd_interp,
        dtdate,
        skymapgreen[1],
        skymapgreen[0],
        interplonvec,
        interplatvec,
        110,
        nshifts=n_shifts,
        background_method=background_method,
        plot=False
    )

    blueimdenoise = np.copy(blueimdenoisewavelet)

    ngreen = (1 / np.std(greenimreg[np.where(~np.isnan(greenimreg))])) ** (6.5 / 8)
    nred = (1 / np.std(redimreg[np.where(~np.isnan(redimreg))])) ** (6.5 / 8)
    nblue = (1 / np.std(blueimreg[np.where(~np.isnan(blueimreg))])) ** (6.5 / 8)

    greenframe = np.copy(greenimreg)
    greenframe[np.where(np.isnan(greenframe))] = greenbgbright

    blueframe = np.copy(blueimreg)
    blueframe[np.where(np.isnan(blueframe))] = bluebgbright

    redframe = np.copy(redimreg)
    redframe[np.where(np.isnan(redframe))] = redbgbright

    greenmin = np.amin(greenframe)
    bluemin = np.amin(blueframe)
    redmin = np.amin(redframe)

    colormat = np.asarray([
        nred * (redframe - redmin),
        ngreen * (greenframe - greenmin),
        nblue * (blueframe - bluemin)
    ]).astype(float)

    maxbright = np.amax(colormat)
    colormat /= maxbright

    greenframe = np.copy(greenimdenoise)
    greenframe[np.where(np.isnan(greenframe))] = greenbgbright

    blueframe = np.copy(blueimdenoise)
    blueframe[np.where(np.isnan(blueframe))] = bluebgbright

    redframe = np.copy(redimdenoise)
    redframe[np.where(np.isnan(redframe))] = redbgbright

    colormat = np.asarray([
        nred * (redframe - redmin),
        ngreen * (greenframe - greenmin),
        nblue * (blueframe - bluemin)
    ]).astype(float)

    colormat /= maxbright

    redray, greenray, blueray = to_rayleighs(
        redimdenoise,
        greenimdenoise,
        blueimdenoise,
        redbgbright,
        greenbgbright,
        bluebgbright
    )

    badrange = np.where(np.isnan(redray + blueray + greenray))
    redray[badrange] = np.nan
    greenray[badrange] = np.nan
    blueray[badrange] = np.nan

    negatives = np.zeros_like(blueray)
    negatives[np.where((redray < 0) | (blueray < 0) | (greenray < 0))] = 1
    negatives[np.where(np.isnan(blueray))] = np.nan

    redray[np.where(redray < 0)] = 0
    greenray[np.where(greenray < 0)] = 0
    blueray[np.where(blueray < 0)] = 0

    print("Decimating images...")

    dec = 2

    redraydec = redray[::dec, ::dec]
    blueraydec = blueray[::dec, ::dec]
    greenraydec = greenray[::dec, ::dec]

    greenframe = np.copy(greenimdenoise)[::dec, ::dec]
    greenframe[np.where(np.isnan(greenframe))] = greenbgbright

    blueframe = np.copy(blueimdenoise)[::dec, ::dec]
    blueframe[np.where(np.isnan(blueframe))] = bluebgbright

    redframe = np.copy(redimdenoise)[::dec, ::dec]
    redframe[np.where(np.isnan(redframe))] = redbgbright

    colormat = np.asarray([
        nred * (redframe - redmin),
        ngreen * (greenframe - greenmin),
        nblue * (blueframe - bluemin)
    ]).astype(float)

    colormat /= maxbright

    print("Calculating Q and E0...")

    qout, e0out, minq, maxq, mine0, maxe0 = calculate_E0_Q_v2(
        redraydec,
        greenraydec,
        blueraydec,
        v,
        minE0=150,
        generous=True
    )

    print("Calculating conductivities given Q and E0...")

    SigP, SigH = calculate_Sig(qout, e0out, v, generous=True)

    maglon_dec = maglon[::dec, ::dec]
    maglat_dec = maglat[::dec, ::dec]

    plt.title('Map of Q in Geomagnetic Coordinates')
    plt.pcolormesh(maglon_dec, maglat_dec, qout, cmap='plasma')
    plt.colorbar(label='mW/m$^2$')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    plt.savefig(os.path.join(group_outdir, 'Q_geomag.png'))
    plt.close()

    plt.title('Map of E0 in Geomagnetic Coordinates')
    plt.pcolormesh(maglon_dec, maglat_dec, e0out, cmap='viridis')
    plt.colorbar(label='eV')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    plt.savefig(os.path.join(group_outdir, 'E0_geomag.png'))
    plt.close()

    plt.title('Map of SigP in Geomagnetic Coordinates')
    plt.pcolormesh(maglon_dec, maglat_dec, SigP, cmap='magma')
    plt.colorbar(label='mho ($\mho$)')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    plt.savefig(os.path.join(group_outdir, 'SigP_geomag.png'))
    plt.close()

    plt.title('Map of SigH in Geomagnetic Coordinates')
    plt.pcolormesh(maglon_dec, maglat_dec, SigH, cmap='cividis')
    plt.colorbar(label='mho ($\mho$)')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    plt.savefig(os.path.join(group_outdir, 'SigH_geomag.png'))
    plt.close()

    return dtdate, group_outdir, maglon_dec, maglat_dec, qout, e0out, SigP, SigH