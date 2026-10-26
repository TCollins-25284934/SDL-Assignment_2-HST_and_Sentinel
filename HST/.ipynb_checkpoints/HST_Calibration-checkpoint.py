import pathlib
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.optimize import curve_fit
from photutils.background import Background2D, MedianBackground

here = pathlib.Path(".")

def load_calibration_files(index, key):
    index = str(index)
    file_stem = "ubai" + index + "m_"
    subfolder = "*/ubai"+ index + "*/*.fits"
    cal_files = list(here.rglob(subfolder))
    stems = [file.stem.replace(file_stem, "").replace("file", "") for file in cal_files]
    new_dict = dict(zip(stems, cal_files))
    return new_dict[key]


def image_extract(filename):

    with fits.open(filename) as hdul:

        for hdu in hdul:
            if (isinstance(hdu.data, np.ndarray) and hdu.data.ndim == 3 and 4 in hdu.data.shape):
                data = hdu.data.astype(float)
                return data

    raise ValueError(f"No suitable image stack found in {filename}")


def apply_mask(raw, mask):
    corrected = raw.copy()

    corrected[mask != 0] = np.nan

    return corrected


def get_blev_values(blev_chip):

    overscan = blev_chip[9:790, 8:14]

    odd_bias = np.mean(overscan[:, ::2])
    even_bias = np.mean(overscan[:, 1::2])

    return odd_bias, even_bias


def apply_blev_correction(raw, blev_file):

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

    return image - bias_cube


def subtract_dark(image, dark_cube, raw_file):
    with fits.open(raw_file) as hdul:
        exptime = hdul[0].header.get("EXPTIME")
    return image - dark_cube * exptime




def apply_flat(image, flat_cube):
    return image * flat_cube


def subtract_sky(image, background):
    return image - background


def calibrate_exposure(index):
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

    image = apply_blev_correction(
        image,
        blev_file
    )

    image = subtract_bias(
        image,
        bias
    )

    image = subtract_dark(
        image,
        dark, raw_file
    )

    image = apply_flat(
        image,
        flat
    )

    sky = get_sky(image)

    for chip in range(image.shape[0]):
        image[chip] -= sky[chip]
    # print(sky)

    # image = subtract_sky(image, sky)
    # image = transform_by('sqrt', image, pmin=1, pmax=99)
    # image = transform_by('log', image, pmin=15, pmax=85)


    return image


def show_chip(image_cube, chip=0):

    fig, ax = plt.subplots(figsize=(8, 8))

    image = image_cube[chip,:, :]
    image = transform_by('sqrt', image, pmin=1, pmax=99)
    ax.imshow(
        image,
        origin="lower"
    )

    ax.set_title(f"Chip {chip}")

    plt.tight_layout()
    plt.show()


def normalize_data(x, pmin=None, pmax=None):
    """
    normalize data between 0 and 1, optionally clipping the original data
    between user-specified min/max values, or percentiles

    Parameters:
    -----------
    x: np.ndarray
        The data to normalize
    dmin: float or None
        The minimum data value to use for normalization
    dmax: float or None)
        The maximum data value to use for normalization
    percentiles: list, tuple, int, or None
        Instead of using manual dmin/dmax values, use percentiles instead

    Returns:
    -----------
    norm: np.ndarray
        data normalized to [0, 1]
    dmin, dmax: float, float
        min/max values used for normalization
    """
    res = np.asarray(x, dtype=np.float64)
    # quick fix: np.percentile will complain if pmin, pmax are None!
    # if no pmin/pmax values are passed, we'll just assume they want
    # the global min and max values
    if not pmin:
        pmin = 0
    if not pmax:
        pmax = 100

    # vmin, vmax = np.percentile(res, [pmin, pmax])
    vmin, vmax = np.nanpercentile(res, [pmin, pmax])
    # basic min-max normalization here
    norm = (res - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)

    # make sure it's definitely between 0 and 1
    return np.clip(norm, 0, 1)


def transform_by(func, x, pmin=None, pmax=None):
    """
    transform/stretch the data by a named function,
    normalizing the data if necessary

    Parameters:
    -----------
    func: str
        one of 'linear', 'sqrt' or 'log'
    x: np.ndarray
        data to transform
    pmin: int, float, or None
        lower percentile to use for min/max normalization
    pmax: int, float, or None
        upper percentile to use for min/max normalization
    Returns:
    ----------
    out : np.array
        the transformed data
    """

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
    """
    compute the simple MAD and clip the data

    Parameters:
    -----------
    sample : np.array
        the sample data to clip
    threshold : int, float
        the number of MADs to apply for clipping
    Returns:
    -----------
    flag : np.ndarray
        a mask of flagged values
    """
    median = np.nanmedian(sample)
    mad = np.nanmedian(np.abs(sample - median))
    stddev = 1.4826 * mad

    # flag = (sample - median) > (threshold * stddev)
    flag = (sample - median) / stddev > threshold
    # print(f"MAD thresholding produced {flag.sum()} values above the threshold")
    return flag

def filter_stddev(sample, threshold=3):
    """
    compute the simple stddev and clip the data

    Parameters:
    -----------
    sample : np.array
        the sample data to clip
    threshold : int, float
        the number of MADs to apply for clipping
    Returns:
    -----------
    flag : np.ndarray
        a mask of flagged values
    """
    mean = np.nanmean(sample)
    stddev = np.nanstd(sample)
    flag = (sample - mean) / stddev > threshold
    # print(f"STDDEV thresholding produced {flag.sum()} values above threshold")
    return flag


def filter_by(data, kind='mad', thresh=3):
    """ a wrapper to call either of the above functions """
    if kind.lower() == 'mad':
        filtered = filter_mad(data, thresh)
    elif kind.lower() in ['std', 'stddev']:
        filtered = filter_stddev(data, thresh)
    return filtered

def get_sky(image):
    skies = []
    for chip in range(image.shape[0]):

        flag = filter_by(image[chip], 'mad', 3)

        masked = np.ma.masked_array(image[chip],mask=flag | np.isnan(image[chip]))


        skies.append(np.ma.median(masked))

    return np.array(skies)

def interpolate_sky(image, box_size):
    skies = []
    for i in range(image.shape[0]):
        flag = filter_by(image[i], 'mad', 3)
        bkg = Background2D(
            image[i],
            box_size=(box_size,box_size),
            mask=flag,
            bkg_estimator=MedianBackground()
        )
        skies.append(bkg.background)

def difference(image,index, chip):
    data_file = 'hst-data/F336W/ubai' + str(index) +'m_c0m_aligned.fits'

    with fits.open(data_file) as hdul:
        reference = hdul[1].data.astype(float)

    mine = image[chip]
    mine = transform_by('sqrt', mine, pmin=1, pmax=99)
    ref_display = transform_by('sqrt', reference, pmin=1, pmax=99)
    # my_display = transform_by('sqrt', mine, pmin=1, pmax=99)

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
    """
    compute and plot a histogram from some given data, correctly
    handling normalization

    Parameters:
    -----------
    ax: matplotlib.axes.Axes instance
        where to plot the data
    x: array
        input values
    bins: int
        number of bins to use
    range: tuple or None
        bin range
    density: bool (default False)
        if True, return a noramlized probability density
    normed: bool, default True
        if True, return a normalized histogram
    histtype: one of 'bar', 'stacked', 'step', 'stepfilled'; default 'step'
        what type of histogram to draw

    Returns:
    ----------
    n, bins, patches:
        see Axes.stairs for more information
    """
    counts, bin_edges = np.histogram(x, bins, range=xrange)
    widths = bin_edges[1:] - bin_edges[:-1]

    if density:
        counts = counts / (widths * x.size)
    elif normed:
        counts = counts / widths

    return ax.hist(bin_edges[:-1], bin_edges, weights=counts, histtype=histtype, **kwargs)

def histogram_overlay(image1, index):
    data_file = 'hst-data/F336W/ubai' + str(index) + 'm_c0m_aligned.fits'

    with fits.open(data_file) as hdul:
        reference = hdul[1].data.astype(float)

    # mine = MAD.transform_by('sqrt', mine, pmin=1, pmax=99)
    # ref_display = MAD.transform_by('sqrt', reference, pmin=1, pmax=99)

    fig, ax = plt.subplots(figsize=(9, 6))
    histogram(ax, image1, bins=250, color='blue', label='Image 1')
    histogram(ax, reference, bins=250, color='red', label='Image 2')
    ax.set_yscale('log')
    plt.grid(ls=':')
    plt.tight_layout()
    plt.show()

def magnitude_difference_histogram(my_mags, ref_mags):

    delta_mag = my_mags - ref_mags

    fig, ax = plt.subplots(figsize=(9, 6))

    histogram(ax, delta_mag, bins=20, color='blue')

    ax.axvline(np.nanmean(delta_mag), color='red', linestyle='--', label=f"Mean = {np.nanmean(delta_mag):.3f}")

    ax.axvline(np.nanmedian(delta_mag), color='green', linestyle='--', label=f"Median = {np.nanmedian(delta_mag):.3f}")

    ax.set_xlabel("Magnitude Difference (Mine - Reference)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Magnitude Differences")

    ax.legend()

    plt.tight_layout()
    plt.show()

def magnitude_residual_plot(my_mags, ref_mags):

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
    stacks = np.asarray(stacks)
    combined_images = np.median(stacks, axis=0)
    show_chip(combined_images,chip)
    return combined_images

def get_bright_sources(image, sigma=1, nsize=10, max_sources=100):
    """
    Find the brightest local maxima in an image.

    Parameters
    ----------
    image : 2D ndarray
        Input image

    sigma : float
        Gaussian smoothing width

    nsize : int
        Maximum filter size

    max_sources : int
        Number of brightest detections to return

    Returns
    -------
    coords : ndarray
        Array of shape (N,2) containing
        (y,x) coordinates
    """

    # image = gaussian_filter(image, sigma=sigma)

    local_max = maximum_filter(image, size=nsize)

    local_max_mask = (local_max == image)

    coords = np.argwhere(local_max_mask)

    print(f"Found {len(coords)} " + f"candidate sources")

    brightness = image[coords[:, 0], coords[:, 1]]

    order = np.argsort(brightness)[::-1]

    coords = coords[order]

    coords = coords[:max_sources]

    print(f"Keeping {len(coords)} " + f"brightest sources")
    return coords

def show_detected_sources(image, coords):

    display = transform_by('sqrt', image, pmin=1, pmax=99)

    fig, ax = plt.subplots(figsize=(8,8))

    ax.imshow(display, origin='lower')

    ax.scatter(coords[:,1], coords[:,0], s=50, facecolors='none', edgecolors='red')

    ax.set_title(f"{len(coords)} detected stars")

    plt.show()

def circular_aperture_flux(image,center,radius=5):
    y0, x0 = center

    y, x = np.indices(image.shape)

    mask = ((x - x0) ** 2 + (y - y0) ** 2) <= radius ** 2

    return np.nansum(image[mask])

def flux_to_mag(flux, photflam, photzpt, zp_corr=0):

    flux_density = flux * photflam

    if flux_density <= 0:
        return np.nan

    return ( -2.5 * np.log10(flux_density) + photzpt + zp_corr)

def gaussian_flux(popt):

    amp, x0, y0, sigx, sigy = popt

    return (2.0 * np.pi * amp * abs(sigx) * abs(sigy))

def compare_magnitudes(image, index, chip=0, radius=5, max_sources=100, box_size=None, photometry="aperture"):
    filename = (f"hst-data/F336W/" + f"ubai{index}m_c0m_aligned.fits")

    with fits.open(filename) as hdul:
        reference = (hdul[1].data.astype(float))
        header = hdul[1].header

        photflam = header["PHOTFLAM"]
        photzpt = header["PHOTZPT"]

        zp_corr = header.get("ZP_CORR", 0)
    if box_size != None:
        reference = square_aperture(reference, center=[400, 400], box_size=box_size)
    margin = 25

    detect_image = image[margin:-margin, margin:-margin]

    coords = get_bright_sources(detect_image, sigma=1, nsize=10, max_sources=max_sources)

    coords[:, 0] += margin
    coords[:, 1] += margin
    # coords = get_bright_sources(image, sigma=1, nsize=10, max_sources = 100)

    show_detected_sources(reference, coords)
    show_detected_sources(image, coords)

    # my_image = image[chip]

    my_mags = []
    ref_mags = []

    half = 25 // 2

    ny, nx = image.shape

    for y, x in coords:

        if (
                y < half or
                y >= ny - half or
                x < half or
                x >= nx - half
        ):
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

            # print(
            #     f"({y:4d}, {x:4d}) "
            #     f"FWHMx={fwhm_x:.2f} "
            #     f"FWHMy={fwhm_y:.2f}"
            # )

        else:

            raise ValueError("photometry must be 'aperture', 'psf_aperture', or 'gaussian'")

        my_mag = flux_to_mag(my_flux, photflam, photzpt, zp_corr)

        ref_mag = flux_to_mag(ref_flux, photflam, photzpt, zp_corr)

        my_mags.append(my_mag)
        ref_mags.append(ref_mag)

    my_mags = np.asarray(my_mags)

    ref_mags = np.asarray(ref_mags)

    magnitude_difference_histogram(my_mags, ref_mags)

    magnitude_residual_plot(my_mags, ref_mags)

    delta_mag = (my_mags - ref_mags)
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

    fig, ax = plt.subplots(figsize=(7,7))

    ax.scatter(ref_mags, my_mags)

    mn = np.nanmin([ref_mags.min(), my_mags.min()])

    mx = np.nanmax([ref_mags.max(), my_mags.max()])

    ax.plot([mn, mx], [mn, mx], 'r--')

    ax.set_xlabel("Reference magnitude")

    ax.set_ylabel("My magnitude")

    ax.set_title( "Magnitude comparison")

    plt.show()

    # return delta_mag
    return None

def estimate_gaussian_params(data):
    """ Use Gaussian moments to estimate initial parameters """
    res = data.copy() # operate on a copy to be safe
    res[res < 0] = 0.
    total = np.sum(res)

    # amplitude is easy
    amp0 = res.max() - res.min()

    # the first moment is the intensity-weighted centroid estimate
    y, x = np.indices(res.shape)
    x0 = np.sum(x * res) / total
    y0 = np.sum(y * res) / total

    # the second moment estimates the peak width
    sigx = np.sqrt(np.sum(((x - x0) ** 2) * res) / total)
    sigy = np.sqrt(np.sum(((y - y0) ** 2) * res) / total)

    return amp0, x0, y0, sigx, sigy

def psf_radius(image, center, box_size=15, radius_factor=3):

    cutout = square_aperture(image, center=center, box_size=box_size)

    popt, pcov = fit_gauss_2d(cutout)

    amp, x0, y0, sigx, sigy = popt

    sigma = np.mean([sigx, sigy])

    radius = radius_factor * sigma

    return radius, popt

def gauss_2d(coords, amp, x0, y0, sigx, sigy):
    """ return a 2d gaussian evaluation """
    y, x = coords
    term1 = (x - x0)**2 / (2 * sigx**2)
    term2 = (y - y0)**2 / (2 * sigy**2)
    g = amp * np.exp(-(term1 + term2))
    return g.ravel()

def fit_gauss_2d(data, init_par=None):
    """ a fitting routine for a 2d gaussian function """

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
    """ get the intensity weighted centroid of an image """
    y, x = np.indices(data.shape)
    x0 = np.sum(x * data) / data.sum()
    y0 = np.sum(y * data) / data.sum()
    return x0, y0

def moment_2(data):
    """ get the second moment to estimate the peak width """
    y, x = np.indices(data.shape)
    x0, y0 = moment_1(data)
    sigx = np.sqrt(np.sum(((x - x0)**2) * data) / data.sum())
    sigy = np.sqrt(np.sum(((y - y0)**2) * data) / data.sum())
    return sigx, sigy



def square_aperture(data, center, box_size):
    """
    Extract a square cutout from the input data

    Parameters:
    -----------
    data : `np.ndarray`
        the input data
    center : list or tuple
        the center of the box to extract, in y-x order
    box_size : int
        the size of the box in pixels

    Returns:
    -----------
    res : `np.ndarray`
        a cutout of the data
    """

    y, x = map(int, center)

    # ensure the box_size is odd
    if box_size % 2 == 0:
        box_size += 1

    half = box_size // 2

    # use slices
    sly = slice(y - half, y + half + 1)
    slx = slice(x - half, x + half + 1)

    return data[sly, slx]


def main():
    calibrated = calibrate_exposure(2505)
    show_chip(calibrated,0)
    show_chip(calibrated,1)
    show_chip(calibrated,2)
    show_chip(calibrated,3)

    data_file = 'hst-data/F336W/ubai2505m_c0m_aligned.fits'

    with fits.open(data_file) as hdul:
        reference = hdul[1].data.astype(float)

    mine = calibrated[0]
    mine = transform_by('sqrt', mine, pmin=1, pmax=99)
    ref_display = transform_by('sqrt', reference, pmin=1, pmax=99)
    # my_display = transform_by('sqrt', mine, pmin=1, pmax=99)

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

if __name__ == "__main__":
    main()