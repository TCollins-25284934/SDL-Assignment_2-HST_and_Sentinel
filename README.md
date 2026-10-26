# Scientific Image Analysis: Hunga Tonga and HST Calibration

## Overview

This repository contains Python tools and Jupyter notebook analysis for two scientific image-processing tasks:

1.**Hubble Space Telescope (HST) image calibration** — processing astronomical images, estimating background levels, detecting stars, and comparing photometric measurements with HST reference products.
2. **Hunga Tonga–Hunga Ha'apai land-area analysis** — measuring changes in the island's land area over time using Google Earth images.
 

## Repository Contents

| File or directory | Description |
|---|---|
| `Sentinel-assignment.ipynb` | Jupyter notebook analysing the land area of Hunga Tonga–Hunga Ha'apai using Google Earth imagery. |
| `Sentinel_analysis.py` | Python module containing all the functions necessary to run the Sentinel notebook. |
| `HST_Calibration.py` | Python module containing functions for HST image calibration, background estimation, source detection, and photometric comparison. |
| `sentinel-data/White_light/` | Directory containing the Google Earth JPEG images used in the land-area analysis. |
| `hst-data/F336W/` | Directory containing aligned HST reference FITS files used for image and photometric comparisons. |
| HST analysis notebook | Jupyter notebook for the HST analysis of the star cluster. |

The data directories must be supplied separately if they are not already included in the repository.


## 1. Hubble Space Telescope Image Calibration

### Objective

The HST calibration module contains functions for correcting astronomical image data and comparing the resulting images and stellar photometry against aligned HST reference products.

The code works with FITS files containing multi-chip image data.

### Calibration Pipeline

The `calibrate_exposure(index)` function applies the following corrections:

1. **Image loading:** locates the raw exposure and associated calibration files, including mask, bias-level, master-bias, dark, and flat-field data.
2. **Bias-level correction:** estimates bias levels from overscan regions, treating odd and even detector columns separately.
3. **Bias subtraction:** subtracts the master bias from the image.
4. **Dark-current correction:** subtracts the dark-current image scaled by the exposure time.
5. **Flat-field correction:** multiplies the image by the flat-field calibration data.
6. **Sky-background subtraction:** estimates a sky level for each chip and subtracts it from the calibrated image.

The module also includes an optional masking function for replacing masked pixels with `NaN` values. In the current calibration function, applying this mask is commented out.

### Background Estimation and Image Inspection

The module provides tools to:

- Estimate the sky background separately for each detector chip.
- Identify outliers using median absolute deviation (MAD) or standard-deviation thresholds.
- Estimate spatially varying backgrounds using `Background2D` and `MedianBackground`.
- Display individual detector chips using linear, square-root, or logarithmic intensity stretches.
- Compare calibrated images with aligned HST reference images.

### Source Detection and Photometry

The module includes functions for identifying bright sources and measuring their fluxes.

The available photometric approaches are:

- **Circular-aperture photometry:** sums pixel values within a circular region around a source.
- **PSF-based aperture photometry:** fits a two-dimensional Gaussian to estimate a source's characteristic size and uses the fitted width to select an aperture radius.
- **Gaussian photometry:** estimates integrated source flux from a fitted two-dimensional Gaussian profile.

Measured fluxes are converted to magnitudes using the reference image's `PHOTFLAM` and `PHOTZPT` header values, with an optional `ZP_CORR` correction.

The module also provides diagnostic plots, including:

- Calibrated and reference image comparisons.
- Image and reference histograms.
- Magnitude-difference distributions.
- Photometric residuals against reference magnitudes.
- Direct comparisons of measured and reference magnitudes.

These tools are intended to assess the agreement between the calibrated images and the reference products.

## Requirements

The code imports the following Python libraries:

- Python 3
- NumPy
- Matplotlib
- SciPy
- Astropy
- Photutils
- Jupyter Notebook

## 2. Hunga Tonga Land-Area Analysis

### Objective

The aim of this analysis is to estimate how the land area of Hunga Tonga–Hunga Ha'apai changes over time using Google Earth images.

Because the images are JPEGs rather than georeferenced scientific images, the pixel scale must be determined from the scale bar in each image. Land must then be distinguished from the surrounding sea using colour-based image segmentation.

### Methodology

**1. Image loading and inspection**

The images are loaded as RGB arrays and displayed to inspect the island, surrounding water, and image quality.

**2. Pixel-scale calibration**

The scale bar is located in the bottom-right corner of each image. A 400 × 400 pixel region is extracted, and bright pixels are used to identify the horizontal scale bar.

The scale bar represents 600 metres. If its measured length is \(L\) pixels, the pixel scale is

\[
s = \frac{600}{L}\ \text{m/pixel}.
\]

The area represented by each pixel is then

\[
A_{\mathrm{pixel}} = s^2\ \text{m}^2.
\]

**3. Land-mask generation**

Three methods are implemented to classify pixels as land or water:

- **Median brightness mask:** classifies pixels brighter than the image's median brightness as land. This provides a baseline but cannot reliably distinguish land from water.
- **Green-blue difference mask:** uses the normalised colour difference

  \[
  D = \frac{G-B}{R+G+B},
  \]

  where \(R\), \(G\), and \(B\) are the red, green, and blue channels. Pixels above a selected threshold are classified as land.

- **HSV mask:** converts the RGB image to hue, saturation, and value channels. Pixels identified as saturated blue or cyan water are excluded from the land mask.

**4. Threshold selection**

Thresholds are explored by plotting masks at different values and examining two-dimensional histograms of colour properties. These histograms help identify regions where land and water pixels have different colour distributions.

**5. Land-area calculation**

The land area is calculated by counting the pixels classified as land and multiplying by the area represented by each pixel:

\[
A_{\mathrm{land}} =
\frac{N_{\mathrm{land}} A_{\mathrm{pixel}}}{10^6}
\ \text{km}^2,
\]

where \(N_{\mathrm{land}}\) is the number of land pixels.

The resulting areas are compared across image dates and segmentation methods.



- 

## Acknowledgements

AI involvement was used in creating some of this repository, namely this readme file. It was also used to provide suggestions for ideas/functions/debugging.

This project uses Google Earth imagery for the Hunga Tonga land-area analysis and Hubble Space Telescope reference products for astronomical image calibration and photometric comparison.

