import numpy as np
from astropy import units as u
from astropy import stats
from astropy.table import Table, Column, MaskedColumn
from photutils import aperture

from app import log


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def photometry(DM, cfg=None):
    '''
    '''
    catalog = cfg['Catalog'].get('catalog')
    star_r = cfg['Photometry'].getfloat('StarApertureRadius')
    stars = DM.stars.get(catalog, [])

    positions = [p for p in zip(stars['Catalog_X'], stars['Catalog_Y'])]
    apertures = aperture.CircularAperture(positions, star_r)
    sky_apertures = aperture.CircularAnnulus(positions, 2*star_r, 3*star_r)

    # Generate Background Subtracted Image for ApertureStats
    image = u.Quantity(DM.ccd.data)
    backsub = u.Quantity(DM.ccd.data) - np.median(DM.ccd.data)

    ## Centroid stars and determine FWHM
    log.info(f'Centroiding catalog stars')
    apstats = aperture.ApertureStats(image, apertures)
    aperture_properties = [('Centroid_X', 'xcentroid'),
                           ('Centroid_Y', 'ycentroid'),
                           ('FWHM', 'fwhm'),
                           ('elongation', 'elongation'),
                           ('peak_value', 'max'),
                           ('orientation', 'orientation')]
    for p in aperture_properties:
        stars.add_column(Column(name=p[0], data=getattr(apstats, p[1])))
    ncent = np.sum(~np.isnan(stars['Centroid_X']))
    log.info(f'  Centroided {ncent} stars')

    # Evaluate WCS from Centroid Offsets
    deltaX = stars['Catalog_X'] - stars['Centroid_X']
    stars.add_column(Column(name='WCSOffsetX', data=deltaX))
    deltaY = stars['Catalog_Y'] - stars['Centroid_Y']
    stars.add_column(Column(name='WCSOffsetY', data=deltaY))
    deltaR = (deltaX**2+deltaY**2)**0.5
    stars.add_column(Column(name='WCSOffsetR', data=deltaR))
    median_offset = np.median(stars['WCSOffsetR'])
    DM.ccd.header['WCSMDOFF'] = median_offset
    log.info(f'  WCS Median Offset = {median_offset:.2f} pix')

    # Record Typical FWHM
    fwhm_values = stars['FWHM'][~np.isnan(stars['FWHM'])]
    fwhm_mean, fwhm_median, fwhm_stddev = stats.sigma_clipped_stats(fwhm_values)
    DM.ccd.header['FWWM'] = fwhm_median
    DM.ccd.header['FWHMSD'] = fwhm_stddev
    log.info(f'  Typical FWHM = {fwhm_median:.1f} pix (stddev = {fwhm_stddev:.1f} pix)')
    elng_values = stars['elongation'][~np.isnan(stars['elongation'])]
    elng_mean, elng_median, elng_stddev = stats.sigma_clipped_stats(elng_values)
    DM.ccd.header['ELONG'] = elng_median
    DM.ccd.header['ELONGSD'] = elng_stddev
    log.info(f'  Typical elongation = {elng_median:.1f} (stddev = {elng_stddev:.1f})')

    ## Perform aperture photometry on stars
    for color in ['R', 'G', 'B']:
        cmask = DM.mask[color]
        log.info(f'Performing {color} aperture photometry on catalog stars')
        star, errs = apertures.do_photometry(image, mask=cmask)
        star = star/DM.exptime
        stars.add_column(Column(name=f'{color}StarSum', data=star))
        star_area = apertures.area_overlap(image, mask=cmask)
        stars.add_column(Column(name=f'{color}StarArea', data=star_area))
        sky, sky_errs = sky_apertures.do_photometry(image, mask=cmask)
        sky = sky/DM.exptime
        stars.add_column(Column(name=f'{color}SkySum', data=sky))
        sky_area = sky_apertures.area_overlap(image, mask=cmask)
        stars.add_column(Column(name=f'{color}SkyArea', data=sky_area))
        stars.add_column(Column(name=f'{color}SkyMean', data=sky/sky_area))

        # Calculate Zero Point Values
        flux = stars[f'{color}StarSum'] - stars[f'{color}StarArea']*stars[f'{color}SkyMean']
        stars.add_column(Column(name=f'{color}StarFlux', data=flux))
        positive_sky = (stars[f'{color}SkyMean'] > 0)
        Npositive_sky = np.sum(positive_sky)
        log.debug(f'  {Npositive_sky} stars pass positive sky check')
        positive_flux = (stars[f'{color}StarFlux'] > 0)
        Npositive_flux = np.sum(positive_flux)
        log.debug(f'  {Npositive_flux} stars pass positive flux check')
        has_flux = ~np.isnan(stars[f'{color}StarFlux'])
        Nhas_flux = np.sum(has_flux)
        log.debug(f'  {Nhas_flux} stars pass flux existance check')
        not_saturated = stars['peak_value'] < float(cfg['Photometry'].get('SaturationThreshold', 60000))
        Nnot_saturated = np.sum(not_saturated)
        log.debug(f'  {Nnot_saturated} stars pass saturation check')
        good_photometry = positive_sky & positive_flux & has_flux & not_saturated
        stars.add_column(Column(name=f'{color}Photometry', data=good_photometry))
        nphot = np.sum(stars[f'{color}Photometry'])
        log.info(f'  Got photometry on {nphot} stars')
        instmag = [-2.5*np.log10(s[f'{color}StarFlux'])\
                   if s[f'{color}Photometry'] else np.nan for s in stars]
        stars.add_column(Column(name=f'{color}InstMag', data=instmag))
        catalog_mag_name = cfg['Catalog'].get(f'{color}mag')
        zp = stars[catalog_mag_name] - stars[f'{color}InstMag']
        stars.add_column(MaskedColumn(name=f'{color}ZeroPoint', data=zp))

        # Record Typical ZeroPoint
        zp_values = stars[f'{color}ZeroPoint'][stars[f'{color}Photometry']]
        zp_mean, zp_median, zp_stddev = stats.sigma_clipped_stats(zp_values)
        DM.ccd.header[f'{color}ZEROPT'] = zp_median
        DM.ccd.header[f'{color}ZEROSD'] = zp_stddev
        log.info(f'  Typical Zero Point for {color} = {zp_median:.2f} mag (stddev = {zp_stddev:.2f} mag)')

        # Look at Zero Point statistics and flag outlier stars
        outliers = abs(stars[f'{color}ZeroPoint']-zp_median) > 5*zp_stddev
        log.debug(f"  Found {np.sum(outliers)} outlier zero points")
        stars.add_column(Column(name=f'{color}Outliers', data=outliers))

        # Estimate Sky Brightness
        area_of_pixel = DM.ccd.wcs.proj_plane_pixel_area().to(u.arcsec**2).value
        sky_mean_values = stars[f'{color}SkyMean']/area_of_pixel
        Msky = -2.5*np.log10(sky_mean_values) + zp_median
        stars.add_column(Column(name=f'{color}SkyMag', data=Msky))
        Msky_mean, Msky_median, Msky_stddev = stats.sigma_clipped_stats(Msky[stars[f'{color}Photometry']])
        DM.ccd.header[f'{color}SKYB'] = Msky_median
        DM.ccd.header[f'{color}SKYBSD'] = Msky_stddev
        log.info(f'  Typical Sky Brightness for {color} = {Msky_median:.2f} mag (stddev = {Msky_stddev:.2f} mag)')

    DM.stars[catalog] = stars
    