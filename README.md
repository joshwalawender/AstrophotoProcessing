# Astrophoto Processing (APP)

## Single Image Processing

Data Model for Single OSC File
- astropy.io.fits.HDUList
    - HDU [PRIMARY]: original raw file
    - HDU [processed]: current full resolution data + standardized header
        - Header: use HISTORY cards?
    - HDU [blue]: blue data downsampled 2x2
    - HDU [green]: green data downsampled 2x2
    - HDU [red]: red data downsampled 2x2
    - HDU [blue_bkg]: blue background downsampled 2x2
    - HDU [green_bkg]: green background downsampled 2x2
    - HDU [red_bkg]: red background downsampled 2x2
    - HDU [blue_var]: blue variance downsampled 2x2
    - HDU [green_var]: green variance downsampled 2x2
    - HDU [red_var]: red variance downsampled 2x2
    - HDU [catalog_<catalogname>]: FITS table of catalog stars
    - HDU [photometry]: FITS table of photometry results

Processing Steps for a Single File
- Ingest and form data model
- Bias subtract file given a master bias (generic)
- Plate Solve (generic, use info from config)
    - May need configuration if multiple strategies are possible
- Pull catalog stars for field (generic, use info from config)
    - Get Gaia DR3: BPmag, Gmag, RPmag
- Perform photometry on stars (generic)
    - ?? Perform background subtraction (photutils.background)
    - Centroid catalog stars and get FWHM (photutils.aperture.ApertureStats)
    - Bin each color down 2x2
        - Perform aperture photometry (photutils.aperture)
            - Perform on non-background subtracted image
            - Use aperture size from config
        - Generate table: starID, Xcentorid, Y centroid, FWHM, Rflux, Rzp, Rsky, Gflux, Gzp, Gsky, Bflux, Bzp, Bsky
        - Calculate results: fraction of stars detected, <Rzp>, sigma_Rzp, <Gzp>, sigma_Gzp, <Bzp>, sigma_Bzp
    - QLP Plots:
        - FWHM map
        - ZP map
- Put results in FITS file: table to FITSTable, results to header

Functions

- Subtract Bias
    - Input: data model, master bias data model
    - Output: data model (bias subtracted processed HDU)
    - QLP:
- Plate Solve
    - Input: data model, config parameters
    - Output: data model (updated processed HDU header)
    - QLP:
- Get Catalog Stars
    - Input: data model
    - Output: data model (add/update catalog_<catalogname> HDU)
    - QLP:
        - PNG: overlay catalog star positions on image
- Photometry:
    - Input: data model
    - Output: data model (add/update photometry HDU)
    - QLP:
        - PNG: plot FWHM vs. radius
        - PNG: image of ZP distribution
        - PNG: histograms of ZP values by color
    
    