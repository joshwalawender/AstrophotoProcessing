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

    plt.rcParams.update({'font.size': 5})
    plt.figure(figsize=(8,6), dpi=100)
    plot_colors = {'R': 'red', 'G': 'green', 'B': 'blue'}
    for c,color in enumerate(datamodel.zero_point.keys()):
        photometry = stars[stars[f'{color}Photometry']]
        good_photometry = stars[stars[f'{color}Photometry'] & ~stars[f'{color}Outliers']]
        outliers = stars[stars[f'{color}Outliers']]

        plt.subplot(3,2,2*c+1)
#         plt.title(f'{color} Zero Point = {datamodel.zero_point[color]:.3f} (std = {datamodel.zero_point_stddev[color]:.3f})')
        binsize = 0.05
        min_bin = np.round(datamodel.zero_point[color]/binsize)*binsize - 10*np.round(datamodel.zero_point_stddev[color]/binsize)*binsize
        max_bin = np.round(datamodel.zero_point[color]/binsize)*binsize + 10*np.round(datamodel.zero_point_stddev[color]/binsize)*binsize
        bins = np.arange(min_bin, max_bin+binsize, binsize)
        plt.hist(good_photometry[f'{color}ZeroPoint'], bins=bins,
                 color=plot_colors[color], alpha=0.8,
                 label=f'{color} Zero Point Values')
        plt.hist(outliers[f'{color}ZeroPoint'], bins=bins, color='k', alpha=0.3,
                 label='5 sigma Outliers')
        plt.axvline(datamodel.zero_point[color], color='k',
                    label=f'{color} Zero Point = {datamodel.zero_point[color]:.3f}')
        plt.axvline(datamodel.zero_point[color]-datamodel.zero_point_stddev[color],
                    color='k', linestyle=':')
        plt.axvline(datamodel.zero_point[color]+datamodel.zero_point_stddev[color],
                    color='k', linestyle=':')
        plt.legend(loc='best')
        plt.ylabel('Number of Stars')
        if c == 2: plt.xlabel('Zero Point (mag)')

        plt.subplot(3,2,2*c+2)
        plt.plot(good_photometry['peak_value'], good_photometry[f'{color}ZeroPoint'],
                 f'{color.lower()}o', ms=4, alpha=0.3, mew=0,
                 label=f'{color} Zero Point Values')
        plt.plot(outliers['peak_value'], outliers[f'{color}ZeroPoint'], 
                 'kx', ms=3, alpha=0.5,
                 label='5 sigma Outliers')
        plt.axhline(datamodel.zero_point[color], color='k', alpha=0.5,
                    label=f'{color} Zero Point = {datamodel.zero_point[color]:.3f}')
        plt.legend(loc='best')
        plt.xscale('log')
        plt.xlim(7e2,8e4)
        if c == 2: plt.xlabel('Peak Pixel Value in Star (ADU)')
        plt.ylabel('Zero Point (mag)')

    # Save PNG
    ext = Path(datamodel.raw_file_name).suffix
    plot_file = Path(datamodel.raw_file_name.replace(ext, '_zp.png'))
    if plot_file.exists(): plot_file.unlink()
    log.info(f"Saving {str(plot_file)}")
    plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
