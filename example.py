from pathlib import Path
import sys
import configparser

import numpy as np
from astropy import units as u
from astropy.table import Table, Column
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import stats

from app import log
from app.SmartEyeTools.find_stack import find_stack
from app.data_models.OSCImage import OSCImage
from app.data_models.ImageList import ImageList
from app.reduce.bias_subtract import bias_subtract
from app.analyze.run_astrometrydotnet import solve_field
from app.analyze.get_catalog import query_vizier, apply_catalog
from app.analyze.photometry import photometry
from app.analyze.derive_scaling import derive_scaling
from app.plots.plot_zeropoints import plot_zeropoints
from app.plots.plot_skybrightness import plot_skybrightness
from app.plots.plot_WCSoffsets import plot_WCSoffsets


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
SIPorder = 1

[Catalog]
catalog = Gaia DR3
GmagLimit = 16
Gmag = Gmag
Rmag = RPmag
Bmag = BPmag

[Photometry]
StarApertureRadius = 10
SaturationThreshold = 60000
''')


##-------------------------------------------------------------------------
## Example Reduction of SmartEye files
##-------------------------------------------------------------------------

# Derive file info from SmartEye file structure and metadata
objectname = 'M78'
data_dir = Path('~/Desktop/SmartEye_2026-02-05/').expanduser()
stack = find_stack(data_dir, objectname)

# raw_files_dir = data_dir / 'Raw'
# raw_files = sorted(stack['RawFiles'])
# working_dir = data_dir / objectname
# master_bias_file = stack['DarkFile']
# master_bias = OSCImage(master_bias_file)
# catalog = cfg['Catalog'].get('catalog')
# 
# raw_file = raw_files[0]
# working_file = working_dir / raw_file.name.replace('.fit', '_processed.fits')
# image = OSCImage(raw_file)
# bias_subtract(image, master_bias=master_bias)
# # Get estimated center from target name
# try:
#     result = Simbad.query_object(objectname)
#     estimated_center = SkyCoord(result['ra'].value[0], result['dec'].value[0],
#                                 unit=(u.deg, u.deg), frame='icrs')
# except:
#     estimated_center = None
# center_coord = solve_field(image, cfg=cfg,
#                            center_coord=estimated_center, search_radius=1)
# reference_catalog = query_vizier(image, cfg=cfg)
# photometry(image, cfg=cfg)
# image.write(working_file)
# image.write_jpg(radius=cfg['Photometry'].getfloat('StarApertureRadius'))
# plot_WCSoffsets(image, cfg=cfg)
# plot_zeropoints(image, cfg=cfg)
# plot_skybrightness(image, cfg=cfg)
# 
# sys.exit(0)



# Image List
log.info('Loading input images to ImageList')
raw_image_dir = data_dir/'Raw'
images = ImageList([rf for rf in stack['RawFiles'][:2]],
                   working_dir=data_dir / objectname,
                   objectname=objectname,
                   masters={'bias': OSCImage(stack['DarkFile'])},
                   cfg=cfg,
                   )
images.process()
images.results.pprint()
