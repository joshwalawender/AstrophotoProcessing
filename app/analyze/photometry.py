#!python3

## Import General Tools
import numpy as np
from astropy import units as u
from astropy import stats
from astropy.table import Table, Column, MaskedColumn
from photutils import aperture

from app import log


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def photometry(datamodel, cfg=None):
    '''
    '''
    catalog = cfg['Catalog'].get('catalog')
    star_r = cfg['Photometry'].getfloat('StarApertureRadius')
    stars = datamodel.stars.get(catalog, [])

    positions = [p for p in zip(stars['Catalog_X'], stars['Catalog_Y'])]
    apertures = aperture.CircularAperture(positions, star_r)
    sky_apertures = aperture.CircularAnnulus(positions, 2*star_r, 3*star_r)
    # Generate Background Subtracted Image for ApertureStats
    full_image = u.Quantity(np.array(datamodel.data))/datamodel.exptime
    image_backsub = u.Quantity(np.array(datamodel.data) - np.median(datamodel.data))/datamodel.exptime
    ## Centroid stars and determine FWHM
    log.info(f'Centroiding catalog stars')
    apstats = aperture.ApertureStats(image_backsub, apertures)
    aperture_properties = [('Centroid_X', 'xcentroid'),
                           ('Centroid_Y', 'ycentroid'),
                           ('FWHM', 'fwhm'),
                           ('elongation', 'elongation'),
                           ('peak_value', 'max'),
                           ('orientation', 'orientation')]
    for p in aperture_properties:
        stars.add_column(Column(name=p[0], data=getattr(apstats, p[1])))

    # Record Typical FWHM
    fwhm_values = stars['FWHM'][~np.isnan(stars['FWHM'])]
    fwhm_mean, fwhm_median, fwhm_stddev = stats.sigma_clipped_stats(fwhm_values)
    datamodel.fwhm = fwhm_median
    datamodel.fwhm_stddev = fwhm_stddev
    log.info(f'  Typical FWHM = {fwhm_median:.1f} pix (stddev = {fwhm_stddev:.1f} pix)')

    ## Perform aperture photometry on stars
    for color in ['R', 'G', 'B']:
        log.info(f'Performing {color} aperture photometry on catalog stars')
        star_area = apertures.area_overlap(full_image,
                                           mask=datamodel.mask[color])
        star, errs = apertures.do_photometry(full_image,
                                             mask=datamodel.mask[color])
        sky_area = sky_apertures.area_overlap(full_image,
                                              mask=datamodel.mask[color])
        sky, sky_errs = sky_apertures.do_photometry(full_image,
                                                    mask=datamodel.mask[color])
        stars.add_column(Column(name=f'{color}StarArea', data=star_area))
        stars.add_column(Column(name=f'{color}StarSum', data=star))
        stars.add_column(Column(name=f'{color}SkyArea', data=sky_area))
        stars.add_column(Column(name=f'{color}SkySum', data=sky))
        stars.add_column(Column(name=f'{color}SkyMean', data=sky/sky_area))

        # Calculate Zero Point Values
        flux = stars[f'{color}StarSum'] - stars[f'{color}StarArea']*stars[f'{color}SkyMean']
        stars.add_column(Column(name=f'{color}StarFlux', data=flux))
        stars.add_column(Column(name=f'{color}Photometry', data=~np.isnan(stars[f'{color}StarFlux'])))
        instmag = -2.5*np.log10(flux)
        stars.add_column(Column(name=f'{color}InstMag', data=instmag))
        catalog_mag_name = cfg['Catalog'].get(f'{color}mag')
        zp = stars[catalog_mag_name] - stars[f'{color}InstMag']
        stars.add_column(MaskedColumn(name=f'{color}ZeroPoint', data=zp))

        # Record Typical ZeroPoint
        zp_values = stars[f'{color}ZeroPoint'][stars[f'{color}Photometry']]
        zp_mean, zp_median, zp_stddev = stats.sigma_clipped_stats(zp_values)
        datamodel.zero_point[color] = zp_median
        datamodel.zero_point_stddev[color] = zp_stddev
        log.info(f'  Typical Zero Point for {color} = {zp_median:.2f} mag (stddev = {zp_stddev:.2f} mag)')

        # Look at Zero Point statistics and flag outlier stars
        outliers = abs(stars[f'{color}ZeroPoint']-zp_median) > 5*zp_stddev
        log.info(f"  Found {np.sum(outliers)} outlier zero points")
        stars.add_column(Column(name=f'{color}Outliers', data=outliers))

    datamodel.stars[catalog] = stars
    