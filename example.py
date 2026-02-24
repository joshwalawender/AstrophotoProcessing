import configparser

import numpy as np

from app.data_models.OSCImage import OSCImage
from app.reduce.bias_subtract import bias_subtract
from app.analyze.run_astrometrydotnet import solve_field
from app.analyze.get_catalog import query_vizier


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

''')
#index-dir = /Volumes/Ohina2External/astrometry_data


##-------------------------------------------------------------------------
## Example Reduction
##-------------------------------------------------------------------------
input_file = '/Volumes/SmartEyeData/SmartEye_2026-02-05/Raw/exp_000187_0005_0011_30sec_0C.fit'
input_file = '~/git/AstrophotoProcessing/exp_000187_0005_0011_30sec_0C.fit'
input_image = OSCImage(input_file)

master_bias_file = '~/git/AstrophotoProcessing/StackDark_00C_30_350.fit'
master_bias = OSCImage(master_bias_file)
bias_subtract(input_image, master_bias=master_bias)
solve_field(input_image, cfg=cfg)
if input_image.center_coord is not None:
    print(input_image.center_coord.to_string('hmsdms', precision=1))
    print(input_image.radius)

catalog = 'Gaia DR3'
query_vizier(input_image, catalog=catalog)
if catalog in input_image.stars.keys():
    print(input_image.stars.get(catalog))
    print(input_image.stars.get(catalog).keys())

input_image.write('test.fits')