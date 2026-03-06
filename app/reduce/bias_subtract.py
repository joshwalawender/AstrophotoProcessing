# from astropy.nddata import CCDData
import ccdproc

from app import log
from app.data_models.OSCImage import OSCImage


##-------------------------------------------------------------------------
## Bias Subtract
##-------------------------------------------------------------------------
def bias_subtract(DM, master_bias=None):
    assert isinstance(DM, OSCImage)
    assert isinstance(master_bias, OSCImage)

    log.info('Subtracting bias')
    bias_subtracted = ccdproc.subtract_bias(DM.data, master_bias.data)
#     bias_subtracted = DM.data.data - master_bias.data.data
    DM.update_data(bias_subtracted,
                   header=[('BIASSUB', True, 'Bias subtracted')],
                   history=['Bias subtracted'])
