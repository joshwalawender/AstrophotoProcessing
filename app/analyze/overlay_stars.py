#!python3

## Import General Tools
from astropy import units as u
from astropy import visualization as vis


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def overlay_stars(datamodel):
    '''Take the catalog stars and WCS and generate a PNG file 
    '''
    assert 'RA_ICRS' in stars.keys()
    assert 'DE_ICRS' in stars.keys()
    
    