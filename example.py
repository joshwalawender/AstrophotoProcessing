from pathlib import Path
import sys
import configparser

import numpy as np

from app.data_models.OSCImage import OSCImage
from app.reduce.bias_subtract import bias_subtract
from app.analyze.run_astrometrydotnet import solve_field
from app.analyze.get_catalog import query_vizier
from app.analyze.photometry import photometry
from app.plots.overlay_stars import overlay_stars
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
GmagLimit = 15

[Photometry]
StarApertureRadius = 6
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
    if image.center_coord is not None:
        print(image.center_coord.to_string('hmsdms', precision=1))
        print(image.radius)
    query_vizier(image, cfg=cfg)
    photometry(image, cfg=cfg)
    image.write('test.fits')
else:
    print('Loading Existing Image')
    image = OSCImage(working_file)

overlay_stars(image, cfg=cfg)
plot_zeropoints(image, cfg=cfg)
