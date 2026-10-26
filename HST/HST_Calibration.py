import pathlib
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.optimize import curve_fit
from photutils.background import Background2D, MedianBackground

here = pathlib.Path(".")


def load_calibration_files(index, key):
    """Find and return a calibration file for a given exposure and file type."""
    index = str(index)
    file_stem = "ubai" + index + "m_"
    subfolder = "*/ubai"+ index + "*/*.fits"
    cal_files = list(here.rglob(subfolder))
    stems = [file.stem.replace(file_stem, "").replace("file", "") for file in cal_files]
    new_dict = dict(zip(stems, cal_files))
    return new_dict[key]


def image_extract(filename):
    """Load a four-chip image stack from a FITS file."""
    with fits.open(filename) as hdul:
        for hdu in hdul:
            if (isinstance(hdu.data, np.ndarray) and hdu.data.ndim == 3 and 4 in hdu.data.shape):
                data = hdu.data.astype(float)
                return data

    raise ValueError(f"No suitable image stack found in {filename}")


def apply_mask(raw, mask):
    """Replace masked pixels with NaN values."""
    corrected = raw.copy()
    corrected[mask != 0] = np.nan
    return corrected


def get_blev_values(blev_chip):
    """Calculate the mean bias levels for odd and even columns."""
    overscan = blev_chip[9:790, 8:14]
    odd_bias = np.mean(overscan[:, ::2])
    even_bias = np.mean(overscan[:, 1::2])
    return odd_bias, even_bias


def apply_blev_correction(raw, blev_file):
    """Subtract the odd and even column bias levels from each chip."""
    corrected = raw.copy()

    with fits.open(blev_file) as hdul:
        if (hdul[0].data is not None and hdul[0].data.ndim == 3 and 4 in hdul[0].data.shape):
            blev_cube = hdul[0].data.astype(float)
            for chip in range(4):
                odd_bias, even_bias = get_blev_values(blev_cube[chip])
                corrected[chip, :, ::2] -= odd_bias
                corrected[chip, :, 1::2] -= even_bias
        else:
            for chip in range(4):
                blev_chip = hdul[chip + 1].data.astype(float)
                odd_bias, even_bias = get_blev_values(blev_chip)
                corrected[chip, :, ::2] -= odd_bias
                corrected[chip, :, 1::2] -= even_bias

    return corrected


def subtract_bias(image, bias_cube):
    """Subtract the master bias from the image."""
    return image - bias_cube


def subtract_dark(image, dark_cube, raw_file):
    """Subtract the exposure-scaled dark current from the image."""
    with fits.open(raw_file) as hdul:
        exptime = hdul[0].header.get("EXPTIME")
    return image - dark_cube * exptime


def apply_flat(image, flat_cube):
    """Apply the flat-field correction to the image."""
    return image * flat_cube


def subtract_sky(image, background):
    """Subtract the estimated sky background from the image."""
    return image - background


def calibrate_exposure(index):
    """Calibrate an exposure by applying the bias, dark, flat and sky corrections."""
    raw_file = load_calibration_files(index, "raw")
    mask_file = load_calibration_files(index, "mask")
    blev_file = load_calibration_files(index, "blev")
    bias_file = load_calibration_files(index, "bias")
    dark_file = load_calibration_files(index, "dark")
    flat_file = load_calibration_files(index, "flat")

    raw = image_extract(raw_file)
    mask = image_extract(mask_file)
    bias = image_extract(bias_file)
    dark = image_extract(dark_file)
    flat = image_extract(flat_file)
    image = raw
    # image = apply_mask(raw, mask)

    image = apply_blev_correction(image, blev_file)
    image = subtract_bias(image, bias)
    image = subtract_dark(image, dark, raw_file)
    image = apply_flat(image, flat)

    sky = get_sky(image)

    for chip in range(image.shape[0]):
        image[chip] -= sky[chip]

    return image


def show_chip(image_cube, chip=0):
    """Display one chip from the image cube using a square-root noramlization."""
    fig, ax = plt.subplots(figsize=(8, 8))
    image = image_cube[chip, :, :]
    image = transform_by('sqrt', image, pmin=1, pmax=99)
    ax.imshow(image, origin="lower")
    ax.set_title(f"Chip {chip}")
    plt.tight_layout()
    plt.show()


def normalize_data(x, pmin=None, pmax=None):
    """Normalize image values to the range 0 to 1 using percentile limits."""
    res = np.asarray(x, dtype=np.float64)

    if not pmin:
        pmin = 0
    if not pmax:
        pmax = 100

    vmin, vmax = np.nanpercentile(res, [pmin, pmax])
    norm = (res - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)

    return np.clip(norm, 0, 1)


def transform_by(func, x, pmin=None, pmax=None):
    """Apply a linear, square-root or logarithmic stretch to image data."""
    x_norm = normalize_data(x, pmin=pmin, pmax=pmax)

    if func.lower() == 'sqrt':
        return np.sqrt(x_norm)
    elif func.lower() == 'log':
        scale = 1000
        log_scaled = np.log10(1 + scale * x_norm) / np.log10(1 + scale)
        return log_scaled
    elif func.lower() == 'linear':
        return x_norm
    else:
        raise ValueError("Unsupported function: ", func)


def filter_mad(sample, threshold=3):
    """Flag values above a threshold based on the median absolute deviation."""
    median = np.nanmedian(sample)
    mad = np.nanmedian(np.abs(sample - median))
    stddev = 1.4826 * mad

    flag = (sample - median) / stddev > threshold
    return flag


def filter_stddev(sample, threshold=3):
    """Flag values above a threshold based on the mean and standard deviation."""
    mean = np.nanmean(sample)
    stddev = np.nanstd(sample)
    flag = (sample - mean) / stddev > threshold
    return flag


def filter_by(data, kind='mad', thresh=3):
    """Apply either MAD-based or standard-deviation-based filtering."""
    if kind.lower() == 'mad':
        filtered = filter_mad(data, thresh)
    elif kind.lower() in ['std', 'stddev']:
        filtered = filter_stddev(data, thresh)
    return filtered


def get_sky(image):
    """Estimate the median sky background for each chip."""
    skies = []

    for chip in range(image.shape[0]):
        flag = filter_by(image[chip], 'mad', 3)
        masked = np.ma.masked_array(
            image[chip],
            mask=flag | np.isnan(image[chip])
        )
        skies.append(np.ma.median(masked))

    return np.array(skies)


def interpolate_sky(image, box_size):
    """Estimate a spatially varying sky background for each chip."""
    skies = []

    for i in range(image.shape[0]):
        flag = filter_by(image[i], 'mad', 3)
        bkg = Background2D(
            image[i],
            box_size=(box_size, box_size),
            mask=flag,
            bkg_estimator=MedianBackground()
        )
        skies.append(bkg.background)


def difference(image, index, chip):
    """Compare a calibrated chip with its aligned HST reference image."""
    data_file = 'hst-data/F336W/ubai' + str(index) + 'm_c0m_aligned.fits'

    with fits.open(data_file) as hdul:
        reference = hdul[1].data.astype(float)

    mine = image[chip]
    mine = transform_by('sqrt', mine, pmin=1, pmax=99)
    ref_display = transform_by('sqrt', reference, pmin=1, pmax=99)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(mine, origin='lower')
    ax1.set_title("My calibration")
    ax2.imshow(ref_display, origin='lower')
    ax2.set_title("HST reference product")
    plt.show()

    difference = mine - ref_display
    plt.imshow(difference, origin="lower")
    plt.colorbar()
    plt.title("Difference")
    plt.show()

    return difference


def histogram(ax, x, bins=100, xrange=None, density=False,
              normed=True, histtype='step', **kwargs):
    """Calculate and plot a histogram with optional normalization."""
    counts, bin_edges = np.histogram(x, bins, range=xrange)
    widths = bin_edges[1:] - bin_edges[:-1]

    if density:
        counts = counts / (widths * x.size)
    elif normed:
        counts = counts / widths

    return ax.hist(
        bin_edges[:-1], bin_edges,
        weights=counts, histtype=histtype, **kwargs
    )


def histogram_overlay(image1, index):
    """Plot histograms of an image and its aligned HST reference."""
    data_file = 'hst-data/F336W/ubai' + str(index) + 'm_c0m_aligned.fits'

    with fits.open(data_file) as hdul:
        reference = hdul[1].data.astype(float)

    fig, ax = plt.subplots(figsize=(9, 6))
    histogram(ax, image1, bins=250, color='blue', label='Image 1')
    histogram(ax, reference, bins=250, color='red', label='Image 2')
    ax.set_yscale('log')
    plt.grid(ls=':')
    plt.tight_layout()
    plt.show()


def magnitude_difference_histogram(my_mags, ref_mags):
    """Plot the distribution of magnitude differences between two images."""
    delta_mag = my_mags - ref_mags
    fig, ax = plt.subplots(figsize=(9, 6))

    histogram(ax, delta_mag, bins=20, color='blue')
    ax.axvline(
        np.nanmean(delta_mag), color='red', linestyle='--',
        label=f"Mean = {np.nanmean(delta_mag):.3f}"
    )
    ax.axvline(
        np.nanmedian(delta_mag), color='green', linestyle='--',
        label=f"Median = {np.nanmedian(delta_mag):.3f}"
    )

    ax.set_xlabel("Magnitude Difference (Mine - Reference)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Magnitude Differences")
    ax.legend()
    plt.tight_layout()
    plt.show()


def magnitude_residual_plot(my_mags, ref_mags):
    """Plot magnitude residuals against the reference magnitudes."""
    delta_mag = my_mags - ref_mags
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ref_mags, delta_mag, s=25)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Reference Magnitude")
    ax.set_ylabel("Magnitude Residual (Mine - Reference)")
    ax.set_title("Photometric Residuals")
    plt.tight_layout()
    plt.show()


def cosmic_ray_removal(*stacks, chip):
    """Combine image stacks using their pixel-wise median."""
    stacks = np.asarray(stacks)
    combined_images = np.median(stacks, axis=0)
    show_chip(combined_images, chip)
    return combined_images


def get_bright_sources(image, sigma=1, nsize=10, max_sources=100):
    """Find and return the coordinates of the brightest local maxima."""
    local_max = maximum_filter(image, size=nsize)
    local_max_mask = (local_max == image)
    coords = np.argwhere(local_max_mask)

    print(f"Found {len(coords)} candidate sources")
    brightness = image[coords[:, 0], coords[:, 1]]
    order = np.argsort(brightness)[::-1]
    coords = coords[order]
    coords = coords[:max_sources]

    print(f"Keeping {len(coords)} brightest sources")
    return coords


def show_detected_sources(image, coords):
    """Display an image with detected source positions marked."""
    display = transform_by('sqrt', image, pmin=1, pmax=99)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(display, origin='lower')
    ax.scatter(
        coords[:, 1], coords[:, 0],
        s=50, facecolors='none', edgecolors='red'
    )
    ax.set_title(f"{len(coords)} detected stars")
    plt.show()


def circular_aperture_flux(image, center, radius=5):
    """Sum the pixel values inside a circular aperture."""
    y0, x0 = center
    y, x = np.indices(image.shape)
    mask = ((x - x0) ** 2 + (y - y0) ** 2) <= radius ** 2
    return np.nansum(image[mask])


def flux_to_mag(flux, photflam, photzpt, zp_corr=0):
    """Convert flux to magnitude using the image calibration values."""
    flux_density = flux * photflam

    if flux_density <= 0:
        return np.nan

    return -2.5 * np.log10(flux_density) + photzpt + zp_corr


def gaussian_flux(popt):
    """Calculate the integrated flux of a fitted two-dimensional Gaussian."""
    amp, x0, y0, sigx, sigy = popt
    return 2.0 * np.pi * amp * abs(sigx) * abs(sigy)


def compare_magnitudes(image, index, chip=0, radius=5, max_sources=100,
                       box_size=None, photometry="aperture"):
    """Measure source magnitudes and compare them with the HST reference."""
    filename = f"hst-data/F336W/ubai{index}m_c0m_aligned.fits"

    with fits.open(filename) as hdul:
        reference = hdul[1].data.astype(float)
        header = hdul[1].header
        photflam = header["PHOTFLAM"]
        photzpt = header["PHOTZPT"]
        zp_corr = header.get("ZP_CORR", 0)

    if box_size != None:
        reference = square_aperture(
            reference, center=[400, 400], box_size=box_size
        )

    margin = 25
    detect_image = image[margin:-margin, margin:-margin]
    coords = get_bright_sources(
        detect_image, sigma=1, nsize=10, max_sources=max_sources
    )

    coords[:, 0] += margin
    coords[:, 1] += margin

    show_detected_sources(reference, coords)
    show_detected_sources(image, coords)

    my_mags = []
    ref_mags = []
    half = 25 // 2
    ny, nx = image.shape

    for y, x in coords:
        if y < half or y >= ny - half or x < half or x >= nx - half:
            continue

        if photometry == "aperture":
            my_flux = circular_aperture_flux(image, (y, x), radius)
            ref_flux = circular_aperture_flux(reference, (y, x), radius)

        elif photometry == "psf_aperture":
            try:
                my_radius, my_fit = psf_radius(image, (y, x), box_size=25)
                ref_radius, ref_fit = psf_radius(reference, (y, x), box_size=25)
            except RuntimeError:
                print(f"Failed fit ({y}, {x})")
                continue

            radius_used = np.mean([my_radius, ref_radius])
            my_flux = circular_aperture_flux(image, (y, x), radius_used)
            ref_flux = circular_aperture_flux(reference, (y, x), radius_used)

        elif photometry == "gaussian":
            try:
                _, my_fit = psf_radius(image, (y, x), box_size=25)
                _, ref_fit = psf_radius(reference, (y, x), box_size=25)
            except RuntimeError:
                print(f"Failed fit ({y}, {x})")
                continue

            my_flux = gaussian_flux(my_fit)
            ref_flux = gaussian_flux(ref_fit)

            amp, x0, y0, sigx, sigy = my_fit
            fwhm_x = 2.355 * sigx
            fwhm_y = 2.355 * sigy
            mean_fwhm = (fwhm_x + fwhm_y) / 2

            if mean_fwhm < 1:
                continue

        else:
            raise ValueError(
                "photometry must be 'aperture', 'psf_aperture', or 'gaussian'"
            )

        my_mag = flux_to_mag(my_flux, photflam, photzpt, zp_corr)
        ref_mag = flux_to_mag(ref_flux, photflam, photzpt, zp_corr)
        my_mags.append(my_mag)
        ref_mags.append(ref_mag)

    my_mags = np.asarray(my_mags)
    ref_mags = np.asarray(ref_mags)

    magnitude_difference_histogram(my_mags, ref_mags)
    magnitude_residual_plot(my_mags, ref_mags)

    delta_mag = my_mags - ref_mags
    worst = np.nanargmax(np.abs(delta_mag))

    print("Worst residual:", delta_mag[worst])
    print("My mag:", my_mags[worst])
    print("Ref mag:", ref_mags[worst])
    print("Coord:", coords[worst])

    y, x = coords[worst]
    cutout = square_aperture(image, (y, x), 51)

    plt.figure(figsize=(6, 6))
    plt.imshow(transform_by('sqrt', cutout), origin='lower')
    plt.colorbar()
    plt.show()

    print("\nPhotometric Comparison")
    print("----------------------")
    print("Mean Δm:", np.nanmean(delta_mag))
    print("Median Δm:", np.nanmedian(delta_mag))
    print("Std Δm:", np.nanstd(delta_mag))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(ref_mags, my_mags)
    mn = np.nanmin([ref_mags.min(), my_mags.min()])
    mx = np.nanmax([ref_mags.max(), my_mags.max()])
    ax.plot([mn, mx], [mn, mx], 'r--')
    ax.set_xlabel("Reference magnitude")
    ax.set_ylabel("My magnitude")
    ax.set_title("Magnitude comparison")
    plt.show()


def estimate_gaussian_params(data):
    """Estimate Gaussian amplitude, centre and widths from image moments."""
    res = data.copy()
    res[res < 0] = 0.
    total = np.sum(res)

    amp0 = res.max() - res.min()
    y, x = np.indices(res.shape)
    x0 = np.sum(x * res) / total
    y0 = np.sum(y * res) / total
    sigx = np.sqrt(np.sum(((x - x0) ** 2) * res) / total)
    sigy = np.sqrt(np.sum(((y - y0) ** 2) * res) / total)

    return amp0, x0, y0, sigx, sigy


def psf_radius(image, center, box_size=15, radius_factor=3):
    """Fit a Gaussian to a source and estimate its aperture radius."""
    cutout = square_aperture(image, center=center, box_size=box_size)
    popt, pcov = fit_gauss_2d(cutout)
    amp, x0, y0, sigx, sigy = popt
    sigma = np.mean([sigx, sigy])
    radius = radius_factor * sigma
    return radius, popt


def gauss_2d(coords, amp, x0, y0, sigx, sigy):
    """Evaluate a two-dimensional Gaussian at the supplied coordinates."""
    y, x = coords
    term1 = (x - x0)**2 / (2 * sigx**2)
    term2 = (y - y0)**2 / (2 * sigy**2)
    g = amp * np.exp(-(term1 + term2))
    return g.ravel()


def fit_gauss_2d(data, init_par=None):
    """Fit a two-dimensional Gaussian to an image cutout."""
    res = data.copy()
    res[res < 0] = 0.

    if init_par is None:
        amp = res.max() - res.min()
        x0, y0 = moment_1(res)
        sigx, sigy = moment_2(res)
        init_par = [amp, x0, y0, sigx, sigy]

    y, x = np.indices(res.shape)
    popt, pcov = curve_fit(
        gauss_2d,
        (y, x),
        res.ravel(),
        p0=init_par
    )

    return popt, pcov


def moment_1(data):
    """Calculate the intensity-weighted centre of an image."""
    y, x = np.indices(data.shape)
    x0 = np.sum(x * data) / data.sum()
    y0 = np.sum(y * data) / data.sum()
    return x0, y0


def moment_2(data):
    """Estimate the Gaussian widths from the second moments of an image."""
    y, x = np.indices(data.shape)
    x0, y0 = moment_1(data)
    sigx = np.sqrt(np.sum(((x - x0)**2) * data) / data.sum())
    sigy = np.sqrt(np.sum(((y - y0)**2) * data) / data.sum())
    return sigx, sigy


def square_aperture(data, center, box_size):
    """Extract a square cutout centred on the specified pixel."""
    y, x = map(int, center)

    if box_size % 2 == 0:
        box_size += 1

    half = box_size // 2
    sly = slice(y - half, y + half + 1)
    slx = slice(x - half, x + half + 1)

    return data[sly, slx]


def main():
    pass


if __name__ == "__main__":
    main()
```