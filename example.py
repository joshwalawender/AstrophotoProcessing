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
SaturationThreshold = 600000
''')


##-------------------------------------------------------------------------
## Example Reduction of SmartEye files
##-------------------------------------------------------------------------

# Derive file info from SmartEye file structure and metadata
objectname = 'M78'
data_dir = Path('~/Desktop/SmartEye_2026-02-05/').expanduser()
stack = find_stack(data_dir, objectname)

raw_files_dir = data_dir / 'Raw'
raw_files = sorted(stack['RawFiles'])
working_dir = data_dir / objectname
master_bias_file = stack['DarkFile']
master_bias = OSCImage(master_bias_file)
catalog = cfg['Catalog'].get('catalog')

raw_file = raw_files[0]
working_file = working_dir / raw_file.name.replace('.fit', '_processed.fits')
image = OSCImage(raw_file)
bias_subtract(image, master_bias=master_bias)
# Get estimated center from target name
try:
    result = Simbad.query_object(objectname)
    estimated_center = SkyCoord(result['ra'].value[0], result['dec'].value[0],
                                unit=(u.deg, u.deg), frame='icrs')
except:
    estimated_center = None
center_coord = solve_field(image, cfg=cfg,
                           center_coord=estimated_center, search_radius=1)
reference_catalog = query_vizier(image, cfg=cfg)
photometry(image, cfg=cfg)
image.write(working_file)
image.write_jpg(radius=cfg['Photometry'].getfloat('StarApertureRadius'))
# plot_WCSoffsets(image, cfg=cfg)
# plot_zeropoints(image, cfg=cfg)
# plot_skybrightness(image, cfg=cfg)

sys.exit(0)


#####

# Image List
# log.info('Loading input images to ImageList')
# raw_image_dir = data_dir/'Raw'
# images = ImageList([rf for rf in stack['RawFiles'][:2]],
#                    working_dir=data_dir / objectname,
#                    masters={'bias': OSCImage(stack['DarkFile'])},
#                    cfg=cfg,
#                    )
# images.process()
# images.results.pprint()
#
# sys.exit(0)




#####


sys.exit(0)

# Create Summary Table of Images
images = Table([Column([], 'RawFile', str),
                Column([], 'RA', str),
                Column([], 'Dec', str),
                Column([], 'WCS Offset', float),
                Column([], 'FWHM', float),
                Column([], 'Elongation', float),
                Column([], 'RZeroPoint', float),
                Column([], 'RZeroPointStdDev', float),
                Column([], 'RSkyBrightness', float),
                Column([], 'RSkyBrightnessStdDev', float),
                Column([], 'RSkyOffset', float),
                Column([], 'RFluxScaling', float),
                Column([], 'GZeroPoint', float),
                Column([], 'GZeroPointStdDev', float),
                Column([], 'GSkyBrightness', float),
                Column([], 'GSkyBrightnessStdDev', float),
                Column([], 'GSkyOffset', float),
                Column([], 'GFluxScaling', float),
                Column([], 'BZeroPoint', float),
                Column([], 'BZeroPointStdDev', float),
                Column([], 'BSkyBrightness', float),
                Column([], 'BSkyBrightnessStdDev', float),
                Column([], 'BSkyOffset', float),
                Column([], 'BFluxScaling', float),
                ])


# Iterate over files and process them
for i,raw_file in enumerate(raw_files):
    working_file = working_dir / raw_file.name.replace('.fit', '_processed.fits')
    log.info(f'-----------------------------------------------------------')
    log.info(f'Processing file {i+1}/{len(raw_files)}: {working_file.name}')
    if i==0:
        image = OSCImage(raw_file)
        bias_subtract(image, master_bias=master_bias)
        center_coord = solve_field(image, cfg=cfg)
        reference_catalog = query_vizier(image, cfg=cfg)
        photometry(image, cfg=cfg)
        image.write(working_file)
        log.info(f'Writing processed file: {working_file.name}')
    else:
        if working_file.exists():
            log.info(f'Found existing file: {working_file.name}')
            image = OSCImage(working_file)
        else:
            image = OSCImage(raw_file)
            bias_subtract(image, master_bias=OSCImage(master_bias_file))
            success = solve_field(image, cfg=cfg, center_coord=center_coord)
            if success is not None:
                apply_catalog(image, reference_catalog, cfg=cfg)
                photometry(image, cfg=cfg)
                log.info(f'Writing processed file: {working_file.name}')
                image.write(working_file)
            else:
                log.warning(f'Image {working_file.name} failed')
    cc = image.center_coord.to_string('hmsdms', sep=':', precision=1)
    row = {'RawFile': raw_file.name,
           'RA': cc.split()[0],
           'Dec': cc.split()[1],
           'WCS Offset': image.WCS_median_offset,
           'FWHM': image.fwhm,
           'Elongation': image.elongation,
           'RZeroPoint': image.zero_point['R'],
           'RZeroPointStdDev': image.zero_point_stddev['R'],
           'RSkyBrightness': image.sky_brightness['R'],
           'RSkyBrightnessStdDev': image.sky_brightness_stddev['R'],
           'RSkyOffset': 0,
           'RFluxScaling': 0,
           'GZeroPoint': image.zero_point['G'],
           'GZeroPointStdDev': image.zero_point_stddev['G'],
           'GSkyBrightness': image.sky_brightness['G'],
           'GSkyBrightnessStdDev': image.sky_brightness_stddev['G'],
           'GSkyOffset': 0,
           'GFluxScaling': 0,
           'BZeroPoint': image.zero_point['B'],
           'BZeroPointStdDev': image.zero_point_stddev['B'],
           'BSkyBrightness': image.sky_brightness['B'],
           'BSkyBrightnessStdDev': image.sky_brightness_stddev['B'],
           'BSkyOffset': 0,
           'BFluxScaling': 0,
          }
    images.add_row(row)

table_file = working_dir / 'images.csv'
if table_file.exists(): table_file.unlink()
images.write(table_file, format='ascii.csv')


reference = 0
reference_raw_file = raw_files[reference]
reference_working_file = working_dir / reference_raw_file.name.replace('.fit', '_processed.fits')
reference_image = OSCImage(reference_working_file)

for i,entry in enumerate(images):
    working_file = working_dir / str(entry['RawFile']).replace('.fit', '_processed.fits')
    image = OSCImage(working_file)
    derive_scaling(image, reference=reference_image, cfg=cfg)
    for c in ['R', 'G', 'B']:
        images[i][f'{c}SkyOffset'] = image.background_offset[c]
        images[i][f'{c}FluxScaling'] = image.flux_scaling[c]

images['RawFile', 'WCS Offset', 'FWHM', 'GZeroPoint', 'GSkyBrightness', 'GSkyOffset', 'GFluxScaling'].pprint()


# image.write_jpg(radius=cfg['Photometry'].getfloat('StarApertureRadius'))
# plot_zeropoints(image, cfg=cfg)
# plot_skybrightness(image, cfg=cfg)

# image.ds9_set('frame 1')
# image.display()
# image.ds9_set('scale log exp 100')
# image.ds9_set('zoom to fit')
# image.regions_from_catalog()

