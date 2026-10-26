"""Functions for measuring the land area of Hunga Tonga-Hunga Ha'apai from Google Earth images."""

import pathlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.colors import rgb_to_hsv, LogNorm

from scipy import ndimage as ndi

import HST_Calibration as hst

here = pathlib.Path('.')

SCALE_BAR_DISTANCE = 600  # length of the Google Earth scale bar in metres
SCALE_BOX_SIZE = 400  # size of the bottom-right crop containing the scale bar, in pixels


def load_jpgs():
    """Return the sorted list of Google Earth (white light) images."""
    white_images = sorted(list((here / 'sentinel-data' / 'White_light').glob('*.jpg')))
    return white_images


def load_image(file):
    """Read an image as an 8-bit RGB array, so that every mask works on a 0-255 scale."""
    img = mpimg.imread(file)

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    return img[:, :, :3]


def image_grid(n_images, ncols=2):
    """Make a grid of axes large enough for n_images and hide any spare axes."""
    nrows = int(np.ceil(n_images / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        tight_layout=True
    )

    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n_images:]:
        ax.axis('off')

    return fig, axes[:n_images]


def plot_images(list_of_images):
    fig, axes = image_grid(len(list_of_images))

    for ax, image_file in zip(axes, list_of_images):
        ax.imshow(load_image(image_file))
        ax.set_title(image_file.stem)
        ax.axis('off')

    plt.show()


def crop_scale_region(img, boxsize=SCALE_BOX_SIZE):
    """Crop the bottom-right corner of the image, where the scale bar sits."""
    height, width = img.shape[:2]

    cropped = hst.square_aperture(
        img,
        center=[height - boxsize // 2, width - boxsize // 2],
        box_size=boxsize
    )

    return cropped


def plot_scale_crops(list_of_images):
    fig, axes = image_grid(len(list_of_images))

    for ax, image_file in zip(axes, list_of_images):
        ax.imshow(crop_scale_region(load_image(image_file)))
        ax.set_title(image_file.stem)

    plt.show()


def measure_scale_bar(cropped, threshold=220):
    """
    Find the scale bar in a cropped image.

    The bar is the row with the most bright (white) pixels. Returns the length
    of the bar in pixels, the x positions of its two ends and the row it sits on.
    """
    gray = cropped.mean(axis=2)

    mask = gray > threshold

    row_counts = mask.sum(axis=1)
    y_bar = np.argmax(row_counts)

    xs = np.where(mask[y_bar])[0]

    if len(xs) == 0:
        raise ValueError("No scale bar found, try lowering the threshold")

    left = xs.min()
    right = xs.max()

    return right - left, left, right, y_bar


def plot_scales(list_of_images):
    """Plot the measured scale bar in each image and return the pixel sizes in metres."""
    fig, axes = image_grid(len(list_of_images))

    pixel_sizes = []

    for ax, image_file in zip(axes, list_of_images):
        cropped = crop_scale_region(load_image(image_file))

        length_pixels, left, right, y_bar = measure_scale_bar(cropped)
        pixel_size = SCALE_BAR_DISTANCE / length_pixels

        ax.imshow(cropped)
        ax.scatter(
            [left, right],
            [y_bar, y_bar],
            c='red',
            s=10, marker='x'
        )
        ax.set_title(
            f"{image_file.stem}\n"
            f"{length_pixels:.0f} px | "
            f"{pixel_size:.3f} m/pixel",
            fontsize=10
        )

        pixel_sizes.append(pixel_size)

    plt.show()

    return pixel_sizes


def median_mask(rgb_image):
    """Land is anything brighter than the median brightness of the image."""
    gray = np.mean(rgb_image, axis=2)

    threshold = np.median(gray)

    return gray > threshold


def green_blue_difference_image(rgb_image):
    """Normalised green-blue difference, (G - B) / (R + G + B)."""
    rgb_image = rgb_image.astype(float)  # uint8 would wrap around when G < B

    numerator = rgb_image[:, :, 1] - rgb_image[:, :, 2]

    denominator = np.sum(rgb_image, axis=-1)

    diff = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=(denominator != 0)
    )

    return diff


def difference_mask(rgb_image, threshold=0.0):
    """Land is where green exceeds blue by more than the threshold."""
    diff = green_blue_difference_image(rgb_image)

    return diff > threshold


def hsv_image(rgb_image):
    rgb_norm = rgb_image / 255

    hsv = rgb_to_hsv(rgb_norm)

    return hsv


def hsv_mask(
        rgb_image,
        hue_threshold=0.45,
        sat_threshold=0.15):
    """Water is saturated and blue/cyan (hue above the threshold), land is everything else."""
    hsv = hsv_image(rgb_image)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]

    water = (
        (hue > hue_threshold)
        &
        (saturation > sat_threshold)
    )

    return ~water


def overlay_masks(
        image_files,
        mask_function,
        **kwargs):
    """Paint the land pixels of each image red."""
    fig, axes = image_grid(len(image_files))

    for ax, image_file in zip(axes, image_files):
        image = load_image(image_file)

        mask = mask_function(
            image,
            **kwargs
        )

        display = image.copy()
        display[mask] = [255, 0, 0]

        ax.imshow(display)
        ax.set_title(image_file.stem)
        ax.axis('off')

    plt.show()


def plot_masked_images(
        image_files,
        mask_function,
        **kwargs):
    """Show only the land pixels of each image, the water is masked out."""
    fig, axes = image_grid(len(image_files))

    for ax, image_file in zip(axes, image_files):
        image = load_image(image_file)

        mask = mask_function(
            image,
            **kwargs
        )

        # imshow ignores the mask of an RGB masked array, so hide the water with an alpha channel
        rgba = np.dstack((image, np.where(mask, 255, 0).astype(np.uint8)))

        ax.imshow(rgba)
        ax.set_title(image_file.stem)
        ax.axis('off')

    plt.show()


def tune_threshold(
        image,
        mask_function,
        parameter,
        values,
        **kwargs):
    """
    Show the land mask for a range of values of one threshold.

    parameter is the name of the keyword argument of mask_function to vary,
    any other thresholds are held fixed through kwargs.
    """
    fig, axes = image_grid(len(values), ncols=3)

    for ax, value in zip(axes, values):
        mask = mask_function(
            image,
            **{parameter: value},
            **kwargs
        )

        display = image.copy()
        display[mask] = [255, 0, 0]

        ax.imshow(display)
        ax.set_title(f"{parameter} = {value:.2f}")
        ax.axis('off')

    plt.show()


def plot_hsv_channels(rgb_image):
    hsv = hsv_image(rgb_image)

    fig, ax = plt.subplots(
        ncols=3,
        figsize=(15, 5)
    )

    ax[0].imshow(hsv[:, :, 0], cmap='hsv')
    ax[0].set_title("Hue")

    ax[1].imshow(hsv[:, :, 1], cmap='viridis')
    ax[1].set_title("Saturation")

    ax[2].imshow(hsv[:, :, 2], cmap='gray')
    ax[2].set_title("Value")

    plt.tight_layout()
    plt.show()


def plot_difference_distribution(rgb_image, threshold=None):
    """2D histogram of the green-blue difference against the brightness of each pixel."""
    diff = green_blue_difference_image(rgb_image)
    brightness = np.mean(rgb_image, axis=2)

    fig, ax = plt.subplots(figsize=(8, 6))

    h = ax.hist2d(
        diff.ravel(),
        brightness.ravel(),
        bins=200,
        cmap='viridis',
        norm=LogNorm()
    )

    if threshold is not None:
        ax.axvline(
            threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"threshold = {threshold:.3f}"
        )
        ax.legend()

    ax.set_xlabel("(G - B) / (R + G + B)")
    ax.set_ylabel("Brightness (mean of R, G, B)")

    fig.colorbar(h[3], ax=ax, label="Pixels")
    plt.show()


def plot_hue_saturation_distribution(
        rgb_image,
        hue_threshold=None,
        sat_threshold=None):
    """2D histogram of hue against saturation, with the thresholds drawn on if given."""
    hsv = hsv_image(rgb_image)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    h = ax.hist2d(
        hue.ravel(),
        sat.ravel(),
        bins=200,
        cmap='viridis',
        norm=LogNorm()
    )

    if hue_threshold is not None:
        ax.axvline(
            hue_threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"hue = {hue_threshold:.2f}"
        )

    if sat_threshold is not None:
        ax.axhline(
            sat_threshold,
            color='orange',
            linestyle='--',
            linewidth=2,
            label=f"saturation = {sat_threshold:.2f}"
        )

    if hue_threshold is not None or sat_threshold is not None:
        ax.legend()

    ax.set_xlabel("Hue")
    ax.set_ylabel("Saturation")

    fig.colorbar(h[3], ax=ax, label="Pixels")
    plt.show()


def land_area(mask, pixel_area):
    """Area of the True pixels in a mask, in the units of pixel_area."""
    return np.sum(mask) * pixel_area


def measure_land_areas(
        image_files,
        pixel_areas,
        mask_function,
        **kwargs):
    """Land area in km^2 of each image for one mask. Each image has its own pixel area in m^2."""
    areas = []

    for image_file, pixel_area in zip(image_files, pixel_areas):
        image = load_image(image_file)

        mask = mask_function(
            image,
            **kwargs
        )

        areas.append(land_area(mask, pixel_area) / 1e6)

    return areas


def land_areas_by_method(image_files, pixel_areas, methods):
    """
    Measure the land area of every image with every mask.

    methods is a dictionary of label: (mask_function, kwargs). Returns the dates
    (the image names) and a dictionary of label: list of areas in km^2.
    """
    dates = [image_file.stem for image_file in image_files]

    areas = {}

    for label, (mask_function, kwargs) in methods.items():
        areas[label] = measure_land_areas(
            image_files,
            pixel_areas,
            mask_function,
            **kwargs
        )

    return dates, areas


def plot_land_areas(dates, areas):
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, values in areas.items():
        ax.plot(
            dates,
            values,
            marker='o',
            label=label
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Land area (km$^2$)")
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def print_area_table(dates, areas):
    labels = list(areas)
    width = max(max(len(label) for label in labels), 8) + 2

    print(f"{'Date':<14}" + "".join(f"{label:>{width}}" for label in labels))

    for i, date in enumerate(dates):
        print(f"{date:<14}" + "".join(f"{areas[label][i]:>{width}.3f}" for label in labels))
