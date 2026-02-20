#!python3

## Import General Tools
from pathlib import Path
from astropy.io import fits


##-------------------------------------------------------------------------
## Bias Subtract
##-------------------------------------------------------------------------
def bias_subtract(datamodel, master_bias=None):
    if type(master_bias) in [str, Path]:
        master_bias = Path(master_bias).absolute()
        assert master_bias.exists()
        master_bias = fits.open(master_bias)
    elif type(master_bias) in [fits.HDUList]:
        pass

    


