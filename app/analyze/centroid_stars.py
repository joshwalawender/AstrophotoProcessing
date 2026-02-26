#!python3

## Import General Tools
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS
from astropy.nddata import NDData, NDDataArray
from photutils import aperture


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def centroid_stars(datamodel, cfg=None):
    '''Take the catalog stars and WCS and generate a PNG file 
    '''
    catalog = cfg['Catalog'].get('catalog')
    star_r = cfg['Photometry'].getfloat('StarApertureRadius')
    stars = datamodel.stars.get(catalog, [])
    coords = SkyCoord(stars['RAJ2000'], stars['DEJ2000'], frame=ICRS,
                      unit=(u.deg, u.deg),
                      obstime=Time(2000, format='decimalyear'))
    apertures = aperture.SkyCircularAperture(coords, star_r*u.arcsec)
    im = u.Quantity(np.array(datamodel.data) - np.median(datamodel.data))
    wcs = datamodel.get_wcs()
    print()
#     print(type(im))
#     print(im.shape)
#     print(type(wcs))
    print(apertures[0])
    
    apstats = aperture.ApertureStats(im, apertures, wcs=wcs)
    print(apstats)