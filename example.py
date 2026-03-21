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
from app.reduce.reproject import reproject
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
catalog = Gaia_DR3
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



# catalog = cfg['Catalog'].get('catalog')
# working_dir = data_dir / objectname
# master_bias_file = stack['DarkFile']
# master_bias = OSCImage(master_bias_file)
# 
# image_file = data_dir / 'Raw' / sorted(stack['RawFiles'])[0]
# working_file = data_dir / objectname / image_file.name
# refim = OSCImage(image_file)
# bias_subtract(refim, master_bias=master_bias)
# # Get estimated center from target name
# try:
#     result = Simbad.query_object(objectname)
#     estimated = SkyCoord(result['ra'].value[0], result['dec'].value[0],
#                                 unit=(u.deg, u.deg), frame='icrs')
# except:
#     estimated = None
# center_coord = solve_field(refim, cfg=cfg, center_coord=estimated, search_radius=1)
# reference_catalog = query_vizier(refim, cfg=cfg)
# photometry(refim, cfg=cfg)
# refim.split_colors()
# refim.write(working_file)
# 
# image_file = data_dir / 'Raw' / sorted(stack['RawFiles'])[40]
# working_file = data_dir / objectname / image_file.name
# im = OSCImage(image_file)
# bias_subtract(im, master_bias=master_bias)
# # Get estimated center from target name
# try:
#     result = Simbad.query_object(objectname)
#     estimated = SkyCoord(result['ra'].value[0], result['dec'].value[0],
#                                 unit=(u.deg, u.deg), frame='icrs')
# except:
#     estimated = None
# center_coord = solve_field(im, cfg=cfg, center_coord=estimated, search_radius=1)
# reference_catalog = query_vizier(im, cfg=cfg)
# photometry(im, cfg=cfg)
# reproject(im, reference_wcs=refim.ccd.wcs)
# im.write(working_file)
# # im.write_jpg(radius=cfg['Photometry'].getfloat('StarApertureRadius'))
# # plot_WCSoffsets(im, cfg=cfg)
# # plot_zeropoints(im, cfg=cfg)
# # plot_skybrightness(im, cfg=cfg)
# 
# sys.exit(0)



# Image List
log.info('Loading input images to ImageList')
raw_image_dir = data_dir/'Raw'
raw_files = [rf for rf in stack['RawFiles']]
working_dir = data_dir / objectname
if working_dir.exists():
	processed_files = [f for f in working_dir.glob('*fits')]
else:
	processed_files = []

if len(processed_files) == len(raw_files):
    images = ImageList(processed_files,
                       working_dir=working_dir,
                       objectname=objectname,
                       cfg=cfg)
    print('Reading results file on disk')
    images.results = Table.read(images.summary_file, format='ascii.csv')
else:    
    images = ImageList(raw_files,
                       working_dir=working_dir,
                       objectname=objectname,
                       masters={'bias': OSCImage(stack['DarkFile'])},
                       cfg=cfg)
    images.process()
    images.set_reference_image('FWHM', op='min')
    images.reproject()
    summary_file = data_dir / objectname / images.summary_file.name
    if summary_file.exists(): summary_file.unlink()
    images.results.write(summary_file, format='ascii.csv')
    images.write_all()

images.add_filter('FWHM < 90%')
images.add_filter('WCSOffset < 0.20')
images.plot_image_quality()
images.plot_photometry()
images.combine()
