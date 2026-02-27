#!python3

## Import General Tools
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from app import log


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def plot_zeropoints(datamodel, cfg=None):
    '''
    '''
    log.info('Plotting distribution of zero point values in image')
    catalog = cfg['Catalog'].get('catalog')
    stars = datamodel.stars.get(catalog, [])

    photometry = stars[stars['Photometry'] == True]
    good_photometry = stars[stars['Photometry'] & ~stars['Outliers']]
    outliers = stars[stars['Outliers'] == True]

    plt.figure(figsize=(8,6))

    plt.subplot(2,1,1)
    plt.title(f'Image Zero Point = {datamodel.zero_point:.3f} (std = {datamodel.zero_point_stddev:.3f})')
    binsize = 0.05
    min_bin = np.round(datamodel.zero_point/binsize)*binsize - 10*np.round(datamodel.zero_point_stddev/binsize)*binsize
    max_bin = np.round(datamodel.zero_point/binsize)*binsize + 10*np.round(datamodel.zero_point_stddev/binsize)*binsize
    bins = np.arange(min_bin, max_bin+binsize, binsize)
    plt.hist(good_photometry['ZeroPoint'], bins=bins, color='green', alpha=0.8,
             label='Zero Point Values')
    plt.hist(outliers['ZeroPoint'], bins=bins, color='red', alpha=0.3,
             label='5 sigma Outliers')
    plt.axvline(datamodel.zero_point, color='k',
                label=f'Zero Point = {datamodel.zero_point:.3f}')
    plt.axvline(datamodel.zero_point-datamodel.zero_point_stddev,
                color='k', linestyle=':')
    plt.axvline(datamodel.zero_point+datamodel.zero_point_stddev, color='k', linestyle=':')
    plt.legend(loc='best')
    plt.ylabel('Number of Stars')

    plt.subplot(2,1,2)
    plt.plot(good_photometry['ZeroPoint'], good_photometry['Gmag'], 'bo', ms=4,
             label='Zero Point Values')
    plt.plot(outliers['ZeroPoint'], outliers['Gmag'], 'rx', ms=4,
             label='5 sigma Outliers')
    plt.legend(loc='best')
    plt.xlabel('Zero Point (mag)')
    plt.ylabel('Gaia G Magnitude')

    # Save PNG
    ext = Path(datamodel.raw_file_name).suffix
    plot_file = Path(datamodel.raw_file_name.replace(ext, '_zp.png'))
    if plot_file.exists(): plot_file.unlink()
    log.info(f"Saving {str(plot_file)}")
    plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
