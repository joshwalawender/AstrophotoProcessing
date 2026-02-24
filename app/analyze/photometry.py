#!python3

## Import General Tools
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAperture


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def aperture_photometry(stars):
    for