# from multiprocessing import Process
import numpy as np
from astropy.wcs import WCS
from ccdproc import wcs_project

from app import log
from app.data_models.OSCImage import OSCImage


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def reproject(DM, reference_wcs=None):
    if not isinstance(DM, OSCImage):
        log.error(f'Input data model is {type(DM)}')
    if not isinstance(reference_wcs, WCS):
        log.error(f'Input WCS is {type(DM)}')
    DM.split_colors()
    log.info(f'Reprojecting color components')
#     pR = Process(target=wcs_project, args=(DM.red, reference_wcs))
#     pG = Process(target=wcs_project, args=(DM.green, reference_wcs))
#     pB = Process(target=wcs_project, args=(DM.blue, reference_wcs))
#     pR.start()
#     pG.start()
#     pB.start()
#     pR.join()
#     pG.join()
#     pB.join()

    log.debug('Reprojecting red image')
    DM.red = wcs_project(DM.red, reference_wcs)
    DM.red.meta['ALIGNED'] = 'True'

    log.debug('Reprojecting green image')
    DM.green = wcs_project(DM.green, reference_wcs)
    DM.green.meta['ALIGNED'] = 'True'

    log.debug('Reprojecting blue image')
    DM.blue = wcs_project(DM.blue, reference_wcs)
    DM.blue.meta['ALIGNED'] = 'True'
