import numpy as np
from apexpy import Apex
import scipy.interpolate
from skimage.restoration import denoise_wavelet, cycle_spin, estimate_sigma
import matplotlib.pyplot as plt
import os


def interpolate_reggrid(im, oldlon, oldlat, newlonvec, newlatvec):
    """
    Purpose: 
        - interpolates an image, given a grid, onto a new regular grid
        - takes the image, the old lat/lon grid, and lat/lon vectors for the new mesh grid
    """
    
    # Masks out NaNs
    lonmasked = np.ma.masked_invalid(oldlon)
    latmasked = np.ma.masked_invalid(oldlat)

    # Pulls out unmasked part of old grid
    longood = lonmasked[~lonmasked.mask]
    latgood = latmasked[~lonmasked.mask]
    
    # Pulls out part of image corresponding to valid lon/lat coords
    imgood = im[~lonmasked.mask]

    # Creates the new mesh grid
    newlat, newlon = np.meshgrid(newlatvec, newlonvec)

    # Interpolates the image onto the new grid
    newimvec = scipy.interpolate.griddata(np.asarray([longood, latgood]).T, imgood, np.asarray([newlon.reshape(-1), newlat.reshape(-1)]).T, method='linear')

    return newimvec.reshape(newlat.shape), newlon, newlat

  
def background_brightness_darkpatches(im, lon, lat, plot = False):
    """
    Purpose: 
        - given an unmapped image, finds dark patches of sky to estimate background brightness
          and gaussian noise level
    """
    
    # A gaussian function
    def gauss(x, A, x0, sigma):
        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
    # First, break the image up into patches:
    rows, cols = im.shape
    
    # Size of patches - they will be of size *step* x *step*
    step = 10
    
    # Indices to iterate through
    rowinds = np.arange(0, rows, step)
    colinds = np.arange(0, cols, step)
    
    # Construct a list of median brightnesses for each valid patch
    medvec = [] # initialize vector keeping track of medians
    for i in rowinds:
        for j in colinds:
            imbin = im[i: i + step + 1, j: j + step + 1] # extract patch
            imbingood = imbin[np.where(~np.isnan((lon + lat)[i: i + step + 1, j: j + step + 1]))] # all the valid brightnesses in the patch
            if len(imbingood)>=((step+1)**2)/2: # discard patch if it is more than half invalid.
                medi = np.median(imbingood) 
                medvec.append(medi)
            else: # otherwise, calculate the median
                continue
    medvec = np.asarray(medvec)


    # Choose a maximum cutoff brightness, using quantiles or a specified absolute brightness
    medbright = np.sort(medvec)[1] # we will have at least two patches kept

    # Construct a list of all the brightness points from patches below the max brightness cutoff
    imvec = [] # vector keeping track of all pixels from patches dimmer than medbright

    #if plot:
        #plt.pcolor(lon, lat, im)
        #plt.title('image brightness')
        #plt.show()

    # Find sufficiently dim patches
    for i in rowinds:
        for j in colinds:
            imbin = im[i: i + step + 1 , j: j + step + 1] # extract patches as before
            imbingood = imbin[np.where(~np.isnan((lon + lat)[i: i + step + 1, j: j + step + 1]))]
            if len(imbingood) < ((step+1)**2)/2: # patch is more than half invalid
                continue
            if np.median(imbingood) <= medbright: # patch is dim enough
                #if plot:
                    # Scatter the patch's pixels
                    #plt.scatter(lon[i:i+step+1,j:j+step+1][np.where(~np.isnan(imbin))].reshape(-1),lat[i:i+step+1,j:j+step+1][np.where(~np.isnan(imbin))].reshape(-1),s=0.5)
                    #plt.title('Patch Pixel Scattering')
                    #plt.show()
                # Add the pixels to imvec
                imvec.extend(list(imbingood))
    imvec = np.asarray(imvec)

    # Bin up the brightnesses to make a histogram
    bins = np.arange(np.amin(imvec), np.amax(imvec) + 1) # choose the bins as integer numbers of counts
    
    # For plotting/initial guesses, points are binned more aggressively so the trend is clearer
    plotbins = np.arange(np.amin(imvec), np.amax(imvec) + 5, 5)
    
    bincenters = [np.mean(bins[i: i + 2]) for i in range(len(bins) - 1)] # centers of bins
    
    # Histogram for fitting
    hist, _ = np.histogram(imvec, bins)
    hist = np.asarray(hist)
    
    # Histogram for plotting/initial guesses
    plothist, _ = np.histogram(imvec, plotbins)
    plothist = np.asarray(plothist)
    
    # An initial guess of the center of the brightness distribution
    centerguess = np.median(imvec)
    sigguess = estimate_sigma(im[np.where(~np.isnan(lon + lat))])
    
    # Plot histograms and initial guess gaussian
    #if plot:
        #plt.scatter(bincenters, plothist)
        #plt.scatter(bincenters, 5*hist, s=2)
        #plt.plot(bins, 5 * gauss(bins, np.amax(plothist) / 5, centerguess, sigguess), '--', linewidth=1)
        #plt.xlim(np.amin(imvec) - 10, np.amax(imvec) + 10)
        #plt.title('Noise Fit')
        #plt.show()
    
    # Initialize binary to keep track of whether the fit was successful
    fitfailed = False
    try:
        [peak, cent, sig] = scipy.optimize.curve_fit(gauss, bincenters, hist, p0=[np.amax(plothist) / 5, centerguess, sigguess])[0]
    except Exception as e:
        print('Fit failed! Failed at {e}')
        # fitfailed = True
        
        # If fit fails, try corners instead for this particular case
        try: 
            cent, sig = background_brightness_corners(im, lon, lat)
        except Exception as e:
            print('Fit failed for corners method at {e}')
            
            # Make a last resort, not-so-great guess
            try:
                cent = centerguess
                sig = sigguess
            except Exception as e:
                print('Fit failed for all methods at {e}')        
                fitfailed = True
            
    # Range of the data
    hrange = bins[-1] - bins[0]
    
    # Sanity check on parameters - they are almost surely wrong if they fail this
    if (sig>(hrange/2)) | ((cent-centerguess)>(hrange/4)):
        print('Fit failed!')
        fitfailed = True

    if fitfailed:
        cent = np.copy(centerguess) # keep the median estimate for background brightness
        sig = sigguess # skimage noise estimate
 
    print('bg=' + str(cent))
    print('sig=' + str(sig))
    
    #if plot:
        # Plot the best guess gaussian
        #if ~fitfailed:
            #plt.plot(bins, 5 * gauss(bins, peak, cent, sig))
            #plt.title('Best Guess Gaussian')
        #plt.show()
        # Plot the image on a balanced colormap where white is the background brightness we found
        # Possibly useful to assess validity of results
        #plt.pcolor(lon, lat, im, vmin = cent - 4 * sig, vmax = cent + 4 * sig, cmap='seismic')
        #plt.colorbar()
        #plt.title('White is Fitted Background Brightness')
        #plt.show()
    
    return cent, sig

     
def background_brightness_corners(im, lon, lat, plot=False):
    """
    Purpose: 
        - given an unmapped image, uses corners of sky to estimate background brightness
          and gaussian noise level
    """
    
    # A gaussian function
    def gauss(x, A, x0, sigma):
        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
    imvec = im[np.where(np.isnan(lon+lat))]
    
    # Bin up the brightnesses to make a histogram
    bins = np.arange(np.amin(imvec), np.amax(imvec) + 1) # choose the bins as integer numbers of counts
    bincenters = [np.mean(bins[i: i + 2]) for i in range(len(bins) - 1)]
    hist, _ = np.histogram(imvec, bins)

    # An initial guess of the peak of the brightness distribution - this is simply the mode of the histogram
    centerguess = bincenters[np.where(hist == np.amax(hist))[0][0]]

    # An initial guess at the width of the distribution, found by assuming it to be gaussian and finding the 
    sigguess = (centerguess - bincenters[np.where(hist >= np.amax(hist)/2)[0][0]])/np.sqrt(np.log(2)) # crossing point of half max on the lower brightness end
    
    #if plot:
        #plt.scatter(bincenters, hist)
        #plt.plot(bins, gauss(bins, np.amax(hist), centerguess, sigguess))
        #plt.xlim(np.amin(imvec)-10, np.amax(imvec) + 10)
        #plt.title('Noise Fit')
        #plt.show()

    # Fit the the histogram to a gaussian curve
    [peak, cent, sig] = scipy.optimize.curve_fit(gauss, bincenters, hist, p0=[np.amax(hist), centerguess, sigguess])[0]
    print('sig=' + str(sig))
    
    #if plot:
        #plt.plot(bins,gauss(bins, peak, cent, sig))
        #plt.title('Best Guess Gaussian')
        #plt.show()
        
        #plt.pcolor(lon, lat, im, vmin = cent - 3 * sig, vmax = cent + 3 * sig, cmap='seismic')
        #plt.colorbar()
        #plt.title('White is Fitted Background Brightness')
        #plt.show()
    
    return cent, sig


def gaussian_denoise_resample(im, date, lon, lat, newmlonvec, newmlatvec, mapalt_km, width_deg, NS_deg=0, background_method='patches', plot=False): 
    """
    Purpose: 
        - resample an image onto a uniform grid and denoise it
        - project a red, green, and a blue image onto the same new uniform grid
        - specify a new uniform grid in mlon, mlat; resample image onto it, footpoint the grid to 110 km to match the convention used
        - denoising is done through gaussian blurring
    Takes in:
        - image
        - date (for Apex footpointing)
        - old geolon grid
        - old geolat grid
        - new maglon vector
        - new maglat vector
        - altitude [km] of old map
        - blur width in degrees lat
        - blur width in degrees lon
        - plot
    """

    # Set the date for apex coordinates
    #A = Apex(date=date)

    # Convert 
    #maglat, maglon = A.convert(lat.reshape(-1), np.mod(lon.reshape(-1), 360), 'geo', 'apex', height = mapalt_km)
    maglat, maglon = apex_convert(lat, lon, 'geo', 'apex', date, height=mapalt_km)
    #maglon = maglon.reshape(lon.shape)
    #maglat = maglat.reshape(lon.shape)
        
    # Interpolate onto regular grid        
    regim, regmaglon, regmaglat = interpolate_reggrid(im, maglon, maglat, newmlonvec, newmlatvec)
    
    # Find background brightness and estimated gaussian noise level
    if background_method=='patches':
        bgbright, sig = background_brightness_darkpatches(im, lon, lat, plot = plot)
    elif background_method=='corners':
        bgbright, sig = background_brightness_corners(im, lon, lat, plot = plot)
        
    # Estimate sigma using skimage
    sigma_est = estimate_sigma(im[np.where(~np.isnan(lon + lat))])
    print('skimage estimated sig=' + str(sigma_est))
    
    # Grid steps for our new footpointed grid - the new grid is very nearly Cartesian in footlat/footlon
    dlon = np.mean(np.diff(regmaglon, axis=0))
    dlat = np.mean(np.diff(regmaglat, axis=1))

    # Denoise
    if width_deg != 0: # longitudinal gaussian width
        regimfill = np.copy(regim)
        
        # Fill in the region of the image that does not map to the sky with the background brightness value
        regimfill[np.where(np.isnan(regim))] = bgbright
        
        # Plot image in magnetic coordinates
        if plot:
            plt.pcolormesh(regmaglon, regmaglat, regimfill)
            plt.title('Image in Mag Coords')
            plt.show()

        # Blur E-W
        regimblur = scipy.ndimage.gaussian_filter1d(regimfill, width_deg / dlon, axis = 0)
        
        if NS_deg != 0: # latitudinal gaussian width
            # Blur N-S
            regimblur = scipy.ndimage.gaussian_filter1d(np.copy(regimblur), NS_deg / dlat, axis = 1)
        regimblur[np.where(np.isnan(regim))] = np.nan
    else: # if EW blurring degrees == 0, returns the resampled image
        regimblur = regim

    # Footpoint our new grid
    #lat110, lon110 = A.convert(regmaglat.reshape(-1), np.mod(regmaglon.reshape(-1), 360), 'apex', 'geo', height = 110)
    #lat110 = lat110.reshape(regmaglat.shape)
    #lon110 = lon110.reshape(regmaglon.shape)
    lat110, lon110 = apex_convert(regmaglat, regmaglon, 'apex', 'geo', date, height=110)
    
    #if plot:
        #plt.pcolor(lon, lat, im, vmin = np.amin(regimblur[np.where(~np.isnan(regimblur))]), vmax = np.amax(regimblur[np.where(~np.isnan(regimblur))]))
        #plt.title('Original Image')
        #plt.show()

        #if width_deg != 0:
            #plt.pcolormesh(lon110, lat110, regimblur)
            #plt.xlim(np.min(lon), np.max(lon))
            #plt.ylim(np.min(lat), np.max(lat))
            #plt.title('Blurred Image')
            #plt.show()
    
    return regimblur, regim, lon110, lat110, regmaglon, regmaglat, bgbright, sig



def wavelet_denoise_resample(im, date, lon, lat, newmlonvec, newmlatvec, mapalt_km, nshifts=50, background_method = 'patches', plot = False):
    """
    Purpose: 
        - denoising is done through Bayesian thresholding of a nearly shift-invariant discrete wavelet transform
        - 'nshifts' effectively parameterizes the shift-invariance, the larger it is set, the longer the function
          takes to run but the more shift invariant the wavelets are (better quality denoising)
    Takes in:
        - image
        - date (for Apex footpointing)
        - old geolon grid
        - old geolat grid
        - new maglon vector
        - new maglat vector
        - altitude [km] of old map
        - nshifts
        - plot
    """
    
    # Set the date for apex coordinates
    #A = Apex(date=date)

    # Convert 
    #maglat, maglon = A.convert(lat.reshape(-1), np.mod(lon.reshape(-1), 360), 'geo', 'apex', height = mapalt_km)
    #maglon = maglon.reshape(lon.shape)
    #maglat = maglat.reshape(lon.shape)
    maglat, maglon = apex_convert(lat, lon, 'geo', 'apex', date, height=mapalt_km)
        
    # Interpolate original image onto regular grid        
    regim, regmaglon, regmaglat = interpolate_reggrid(im, maglon, maglat, newmlonvec, newmlatvec)
    
    # Find background brightness and estimated gaussian noise level
    if background_method=='patches':
        bgbright, sig = background_brightness_darkpatches(im, lon, lat, plot = plot)
    elif background_method=='corners':
        bgbright, sig = background_brightness_corners(im, lon, lat, plot = plot)

    # Estimate sigma using skimage
    sigma_est = estimate_sigma(im[np.where(~np.isnan(lon + lat))])
    print('skimage estimated sig=' + str(sigma_est))

    # Grid steps for our new footpointed grid - the new grid is very nearly Cartesian in footlat/footlon
    dlon = np.mean(np.diff(regmaglon, axis = 0))
    dlat = np.mean(np.diff(regmaglat, axis = 1))

    # Denoise
    imdenoise = cycle_spin(im, func = denoise_wavelet, max_shifts = nshifts)
    
    # Interpolate denoised image onto regular grid
    regimdenoise, _, _ = interpolate_reggrid(imdenoise, maglon, maglat, newmlonvec, newmlatvec)
    ridnfill = np.copy(regimdenoise)
    
    # Fill in the region of the image that does not map to the sky with the background brightness value
    ridnfill[np.where(np.isnan(regimdenoise))] = bgbright
    
    # Perform small gaussian blur
    regimblur = scipy.ndimage.gaussian_filter1d(ridnfill, 0.1 / dlon, axis = 0)
    regimblur = scipy.ndimage.gaussian_filter1d(np.copy(regimblur), 0.01 / dlat, axis = 1)
    regimblur[np.where(np.isnan(regimdenoise))] = np.nan

    # Footpoint our new grid
    #lat110, lon110 = A.convert(regmaglat.reshape(-1), np.mod(regmaglon.reshape(-1), 360), 'apex', 'geo', height = 110)
    #lat110 = lat110.reshape(regmaglat.shape)
    #lon110 = lon110.reshape(regmaglon.shape)
    lat110, lon110 = apex_convert(regmaglat, regmaglon, 'apex', 'geo', date, height=110)
    
    #if plot:
        #plt.pcolor(lon, lat, im, vmin = np.amin(regimblur[np.where(~np.isnan(regimblur))]), vmax = np.amax(regimblur[np.where(~np.isnan(regimblur))]))
        #plt.title('Original Image')
        #plt.show()

        #plt.pcolormesh(lon110, lat110, regimblur)
        #plt.xlim(np.min(lon), np.max(lon))
        #plt.ylim(np.min(lat), np.max(lat))
        #plt.title('Denoised Image')
        #plt.show()

    return regimblur, regim, lon110, lat110, regmaglon, regmaglat, bgbright, sig

    
def to_rayleighs(redcutin,greencutin,bluecutin,redbg,greenbg,bluebg):
    """
    Purpose: 
        - uses radioactive source calibrations to convert counts to rayleighs
        - specific to the Poker DASC
    """
    
    # Divide by integration time to get counts per second
    redcut = (np.copy(redcutin) - redbg) / 1.5
    greencut = (np.copy(greencutin) - greenbg) / 1
    bluecut = (np.copy(bluecutin) - bluebg) / 1
    
    # Conversion factors, rayleighs / (counts/second)
    redcut *= 23.8
    greencut *= 24.2
    bluecut *= 69.8
    
    return redcut, greencut, bluecut


def apex_convert(lat, lon, source, dest, date, height=0):
    A = Apex(date=date)
    lat_flat = lat.reshape(-1)
    lon_flat = np.mod(lon.reshape(-1), 360)
    ids = np.where(~np.isnan(lat_flat))
    lat_in = lat_flat[ids]
    lon_in = lon_flat[ids]
    if np.shape(height) == ():
        height_in = height
    else:
        height_flat = height.reshape(-1)
        height_in = height_flat[ids]

    lat_out, lon_out = A.convert(lat_in, lon_in, source, dest, height=height_in)
    lat_out_nans = np.empty(lat_flat.shape)
    lon_out_nans = np.empty(lat_flat.shape)
    lat_out_nans[:] = np.nan
    lon_out_nans[:] = np.nan
    lat_out_nans[ids] = lat_out
    lon_out_nans[ids] = lon_out
    return lat_out_nans.reshape(lat.shape), lon_out_nans.reshape(lon.shape)
