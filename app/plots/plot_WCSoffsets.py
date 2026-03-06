from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from astropy.coordinates import Angle
from astropy import units as u

from app import log


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def plot_WCSoffsets(DM, cfg=None):
    log.info('Plotting distribution of WCS/Centroid offset values in image')
    catalog = cfg['Catalog'].get('catalog')
    stars = DM.stars.get(catalog, [])

    # Remove Outliers from Plot
    out = stars['WCSOffsetR'] > 5*DM.WCS_median_offset

    # Determine Angles
    angles = np.tan(-stars['WCSOffsetX']/stars['WCSOffsetY'])
    angles = Angle(angles*u.rad)
    angles.wrap_at(360*u.deg, inplace=True)
    # Determine angle of star within FoV
    dx = stars['Catalog_X']-DM.data.shape[1]/2
    dy = stars['Catalog_Y']-DM.data.shape[0]/2
    PA = np.tan(-dx/dy)
    PA = Angle(PA*u.rad)
    PA.wrap_at(360*u.deg, inplace=True)

    plt.rcParams.update({'font.size': 5})
    plt.figure(figsize=(5,8), dpi=100)

    plt.subplot(3,2,(1,4))
    plt.plot(stars['Catalog_X'][~out], stars['Catalog_Y'][~out], 'bo',
             ms=1, alpha=0.5)
    plt.quiver(stars['Catalog_X'][~out], stars['Catalog_Y'][~out],
               stars['WCSOffsetX'][~out], stars['WCSOffsetY'][~out],
               angles='xy', scale_units='xy', scale=0.004,
               alpha=0.75)
    plt.gca().set_aspect('equal')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

    plt.subplot(3,2,5)
    plt.hist(angles[~out].to(u.deg), bins=36)
    plt.xlabel('PA of WCS Offset (deg)')
    plt.ylabel('N stars')

    plt.subplot(3,2,6)
    plt.plot(PA[~out].to(u.deg), angles[~out].to(u.deg), 'bo', ms=1, alpha=0.5)
    plt.xlabel('PA of WCS Offset (deg)')
    plt.xlabel('PA of Star in FoV (deg)')


    # Save PNG
    ext = Path(DM.raw_file_name).suffix
    plot_file = Path(DM.raw_file_name.replace(ext, '_wcs.png'))
    if plot_file.exists(): plot_file.unlink()
    log.info(f"Saving {str(plot_file)}")
    plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
