from pathlib import Path
import sys
import configparser

import numpy as np

from app import log
from app.data_models.OSCImage import OSCImage
from app.reduce.bias_subtract import bias_subtract
from app.analyze.run_astrometrydotnet import solve_field
from app.analyze.get_catalog import query_vizier
from app.analyze.photometry import photometry
from app.plots.plot_zeropoints import plot_zeropoints


##-------------------------------------------------------------------------
## Define Configuration
##-------------------------------------------------------------------------
cfg = configparser.ConfigParser()
cfg.read_string('''
[Settings]
cleanup = False

[Telescope]
FocalLength = 814
Aperture = 102

[Camera]
PixelSize = 3.76

[Astrometry.net]
solve-field = /opt/homebrew/bin/solve-field
downsample = 2
SIPorder = 4

[Catalog]
catalog = Gaia DR3
GmagLimit = 16
Gmag = Gmag
Rmag = RPmag
Bmag = BPmag

[Photometry]
StarApertureRadius = 9
SaturationThreshold = 60000
''')
#index-dir = /Volumes/Ohina2External/astrometry_data


##-------------------------------------------------------------------------
## Example Reduction
##-------------------------------------------------------------------------
# raw_file = '/Volumes/SmartEyeData/SmartEye_2026-02-05/Raw/exp_000187_0005_0011_30sec_0C.fit'
raw_file = '~/git/AstrophotoProcessing/exp_000187_0005_0011_30sec_0C.fit'
master_bias_file = '~/git/AstrophotoProcessing/StackDark_00C_30_350.fit'
working_file = Path('~/git/AstrophotoProcessing/test.fits').expanduser()
catalog = cfg['Catalog'].get('catalog')

if not working_file.exists():
    image = OSCImage(raw_file)
    bias_subtract(image, master_bias=OSCImage(master_bias_file))
    solve_field(image, cfg=cfg)
    full_catalog = query_vizier(image, cfg=cfg)
    photometry(image, cfg=cfg)
    image.write(working_file)
else:
    image = OSCImage(working_file)

image.ds9_set('frame 1')
image.display()
image.ds9_set('scale log exp 100')
image.ds9_set('zoom to fit')
image.regions_from_catalog()

# image.write_jpg(radius=cfg['Photometry'].getfloat('StarApertureRadius'))
# plot_zeropoints(image, cfg=cfg)
