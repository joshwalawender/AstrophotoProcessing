#!python3

## Import General Tools
import datetime
import ccdproc

from app.data_models.OSCImage import OSCImage


##-------------------------------------------------------------------------
## Bias Subtract
##-------------------------------------------------------------------------
def bias_subtract(datamodel, master_bias=None):
    assert isinstance(datamodel, OSCImage)
    assert isinstance(master_bias, OSCImage)

    bias_subtracted = ccdproc.subtract_bias(datamodel.data, master_bias.data)

    # Update data model
    now = datetime.datetime.now()
    nowstr = now.strftime('%Y-%m-%D %H:%M:%S')
    datamodel.data = bias_subtracted
    datamodel.hdulist[1].header.set('BIASSUB', True, 'Bias subtracted')
    datamodel.hdulist[1].header.add_history(f'Bias subtracted at {nowstr}')
