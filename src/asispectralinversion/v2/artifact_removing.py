import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator
from scipy.ndimage import gaussian_filter

def remove_artifacts(image, method='linear'):
    fig, ax = plt.subplots()
    ax.imshow(image)

    artifacts = []
    circles = []
    
    # DRAW CIRCLES
    def onclick(event):
        if event.inaxes != ax:
            return
        artifacts.append((event.xdata, event.ydata))
        if len(artifacts) % 2 == 0:
            circle = plt.Circle(artifacts[-2], radius=np.linalg.norm(np.array(artifacts[-2]) - np.array(artifacts[-1])), fill=False, color='r')
            ax.add_artist(circle)
            circles.append(circle)
        fig.canvas.draw()

    def on_done(event):
        plt.close()

    def on_restart(event):
        global artifacts, circles
        artifacts = []

        # RESET if need to redraw around artifact
        for circle in circles:
            circle.remove()
        circles = []

        ax.imshow(image)
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)

    done_button = Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Done')
    done_button.on_clicked(on_done)

    restart_button = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Restart')
    restart_button.on_clicked(on_restart)

    plt.show()

    # INTERP
    mask = np.ones_like(image, dtype=bool)
    for i in range(0, len(artifacts), 2):
        center, edge = artifacts[i], artifacts[i+1]
        radius = np.linalg.norm(np.array(center) - np.array(edge))
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        mask[mask_circle] = False

        # largest square inside the circle
        side_length = radius * np.sqrt(2)  # side of inner circle
        half_side = int(side_length / 2)
        center_x, center_y = int(center[0]), int(center[1])
        x_start = max(0, center_x - half_side)
        x_end = min(image.shape[1], center_x + half_side)
        y_start = max(0, center_y - half_side)
        y_end = min(image.shape[0], center_y + half_side)

        # Square mask
        square_mask = np.ones_like(image, dtype=bool)
        square_mask[y_start:y_end, x_start:x_end] = False

        # Combine masks
        combined_mask = mask & square_mask

    # NaN out masked area
    image_copy = image.astype(float).copy()
    image_copy[~combined_mask] = np.nan

    y, x = np.indices(image.shape)
    valid_points = np.where(combined_mask)
    valid_x, valid_y = valid_points[1], valid_points[0]
    valid_values = image_copy[valid_points]

    if method == 'nearest':
        interpolator = NearestNDInterpolator(np.column_stack((valid_x, valid_y)), valid_values)
    elif method == 'linear':
        interpolator = LinearNDInterpolator(np.column_stack((valid_x, valid_y)), valid_values)
    elif method == 'cubic':
        interpolator = CloughTocher2DInterpolator(np.column_stack((valid_x, valid_y)), valid_values)

    interpolated_values = interpolator(x[~mask], y[~mask]) #Interpolating only circle area not whole image
    image_copy[~mask] = interpolated_values

    # Border mask
    border_mask = np.zeros_like(mask)
    border_mask[~mask] = 1
    border_mask = gaussian_filter(border_mask.astype(float), sigma=2)

    # Blend images
    blended = image_copy * border_mask + image * (1 - border_mask)
    blended = np.nan_to_num(blended, nan=np.nanmean(blended))

    return blended




