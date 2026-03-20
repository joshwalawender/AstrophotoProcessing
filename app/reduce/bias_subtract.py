import numpy as np
from astropy.nddata import CCDData
# import ccdproc

from app import log
from app.data_models.OSCImage import OSCImage


##-------------------------------------------------------------------------
## Bias Subtract
##-------------------------------------------------------------------------
def bias_subtract(DM, master_bias=None):
    assert isinstance(DM, OSCImage)
    assert isinstance(master_bias, OSCImage)

    log.info('Subtracting bias')
    # Need to force data in to float to avoid uint wrapping for negative values
    zeros = np.zeros(DM.ccd.shape, dtype=float)
    image = DM.ccd.subtract(CCDData(zeros, unit='adu'))
#     bias_subtracted = ccdproc.subtract_bias(image, master_bias.data)
    DM.ccd.data = image.data - master_bias.ccd.data
    DM.ccd.header['BIASSUB'] = True

#     DM.update_data(bias_subtracted,
#                    header=[('BIASSUB', True, 'Bias subtracted')],
#                    history=['Bias subtracted'])
