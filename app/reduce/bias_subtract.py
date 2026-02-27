#!python3

## Import General Tools
import datetime
import ccdproc
from astropy.io import fits

from app import log
from app.data_models.OSCImage import OSCImage


##-------------------------------------------------------------------------
## Bias Subtract
##-------------------------------------------------------------------------
def bias_subtract(datamodel, master_bias=None):
    assert isinstance(datamodel, OSCImage)
    assert isinstance(master_bias, OSCImage)

    log.info('Subtracting bias')
    bias_subtracted = ccdproc.subtract_bias(datamodel.data, master_bias.data)
    datamodel.update_data(bias_subtracted,
                          header=[('BIASSUB', True, 'Bias subtracted')],
                          history=['Bias subtracted'])
