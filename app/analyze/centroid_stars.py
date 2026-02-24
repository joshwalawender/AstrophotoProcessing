#!python3

## Import General Tools
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS
from photutils import aperture


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def centroid_stars(datamodel, catalog='Gaia DR3'):
    '''Take the catalog stars and WCS and generate a PNG file 
    '''
    stars = datamodel.stars.get(catalog, [])
    sc = SkyCoord(stars['RAJ2000'], stars['DEJ2000'], frame=ICRS,
                  obstime=Time(2000, format='decimalyear'))
