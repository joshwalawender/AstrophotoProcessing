import numpy as np
from astropy.wcs import WCS
from ccdproc import wcs_project

from app import log
from app.data_models.OSCImage import OSCImage


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def reproject(DM, reference_wcs=None):
    assert isinstance(DM, OSCImage)
    assert isinstance(reference_wcs, WCS)

    DM.split_colors()

    log.info('Reprojecting red image')
    DM.red = wcs_project(DM.red, reference_wcs)
    log.info('Reprojecting green image')
    DM.green = wcs_project(DM.green, reference_wcs)
    log.info('Reprojecting blue image')
    DM.blue = wcs_project(DM.blue, reference_wcs)
