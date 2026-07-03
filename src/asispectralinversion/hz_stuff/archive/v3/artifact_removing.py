import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, LassoSelector
from matplotlib.path import Path
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.interpolate import griddata
from skimage.restoration import inpaint_biharmonic
from skimage.filters import gaussian
"""
Some notes:
    - used to be that you select a center point and a radius, but now directly draw on image with cursor to select region of artifact
    - used to be nearest neighbor interpolation, but changed to cubic to have better blurring and less blocky image in the masked region
"""

def remove_artifacts(image, smoothing_sigma=1):
    img = image.astype(float).copy()
    vmin, vmax = np.percentile(img, 1), np.percentile(img, 99)
    fig, ax = plt.subplots()
    img_display = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.subplots_adjust(bottom=0.2)
    mask = np.zeros_like(img, dtype=bool)

    def onselect(verts):
        nonlocal mask
        path = Path(verts)
        y, x = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
        coords = np.vstack((x.ravel(), y.ravel())).T
        contained = path.contains_points(coords).reshape(img.shape)
        mask |= contained
        ax.imshow(mask, cmap='Reds', alpha=0.4)
        fig.canvas.draw()

    lasso = LassoSelector(ax, onselect)

    def on_done(event):
        lasso.disconnect_events()
        plt.close()

    def on_restart(event):
        nonlocal mask
        mask[:] = False
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        fig.canvas.draw()

    done_button = Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Done')
    done_button.on_clicked(on_done)
    restart_button = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Restart')
    restart_button.on_clicked(on_restart)
    plt.show()

    if not mask.any():
        print("No region selected. Returning original image.")
        return img

    # Interpolate over masked region - nearest neighbor
    #known_mask = ~mask
    #known_yx = np.column_stack(np.nonzero(known_mask))
    #known_values = img[known_mask]
    #interpolator = NearestNDInterpolator(known_yx[:, ::-1], known_values)
    #masked_yx = np.column_stack(np.nonzero(mask))
    #interpolated = interpolator(masked_yx[:, ::-1])
    #img[mask] = interpolated

    # Smoothing
    #edge_dist = distance_transform_edt(mask)
    #blend_band = (edge_dist <= smoothing_sigma * 3)
    #blend_weights = np.clip((smoothing_sigma * 3 - edge_dist) / (smoothing_sigma * 3), 0, 1)
    #blurred = gaussian_filter(img, sigma=smoothing_sigma)
    #final_img = img * (1 - blend_weights) + blurred * blend_weights
    
    # Interpolate using biharmonic/inpainting
    ys, xs = np.nonzero(~mask)
    masked_ys, masked_xs = np.nonzero(mask)
    img[mask] = griddata((ys, xs), img[~mask], (masked_ys, masked_xs), method='cubic', fill_value=np.nan)

    img_inpaint = inpaint_biharmonic(img, mask, channel_axis=None)
    final_img = gaussian(img_inpaint, sigma=smoothing_sigma, preserve_range=True)

    return final_img
