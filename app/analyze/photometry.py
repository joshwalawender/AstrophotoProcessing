#!python3

## Import General Tools
import numpy as np
from astropy import units as u
from astropy import stats
from astropy.table import Table, Column
from photutils import aperture


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
    im = u.Quantity(np.array(datamodel.data) - np.median(datamodel.data))/datamodel.exptime
    ## Centroid stars and determine FWHM
    apstats = aperture.ApertureStats(im, apertures)
    aperture_properties = [('Centroid_X', 'xcentroid'),
                           ('Centroid_Y', 'ycentroid'),
                           ('FWHM', 'fwhm'),
                           ('elongation', 'elongation'),
                           ('peak_value', 'max'),
                           ('orientation', 'orientation')]
    for p in aperture_properties:
        stars.add_column(Column(name=p[0], data=getattr(apstats, p[1])))

    ## Perform aperture photometry on stars
    photometry = aperture.aperture_photometry(im, apertures)
    print(photometry[100:140])
    stars.add_column(Column(name='StarArea', data=[apertures.area]*len(stars)))
    stars.add_column(Column(name='StarSum', data=photometry['aperture_sum']))
#     stars.add_column(Column(name='StarSumErr', data=photometry['aperture_sum_err']))

    ## Perform aperture photometry on sky annulus
    sky_apertures = aperture.CircularAnnulus(positions, 2*star_r, 3*star_r)
    sky = aperture.aperture_photometry(im, sky_apertures)
    stars.add_column(Column(name='SkyArea', data=[sky_apertures.area]*len(stars)))
    stars.add_column(Column(name='SkySum', data=sky['aperture_sum']))
#     stars.add_column(Column(name='SkySumErr', data=sky['aperture_sum_err']))

    flux = stars['StarSum'] - stars['StarArea']/stars['SkyArea']*stars['SkySum']
    stars.add_column(Column(name='StarFlux', data=flux))
    instmag = -2.5*np.log10(flux)
    stars.add_column(Column(name='InstMag', data=instmag))
    zp = stars['Gmag'] - stars['InstMag']
    stars.add_column(Column(name='ZeroPoint', data=zp))
    wphot = ~np.isnan(stars['StarFlux'])
    stars.add_column(Column(name='Photometry', data=wphot))

    zp_values = stars['ZeroPoint'][stars['Photometry'] == True]
    zp_mean, zp_median, zp_stddev = stats.sigma_clipped_stats(zp_values)
    datamodel.zero_point = zp_median
    datamodel.zero_point_stddev = zp_stddev

    datamodel.stars[catalog] = stars
    