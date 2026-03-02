#!python3

from pathlib import Path
import numpy as np
import scipy
from astropy import units as u
from astropy.nddata import CCDData, block_reduce
from astropy import visualization as vis
from matplotlib import pyplot as plt

from app import log


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def plot_skybrightness(DM, cfg=None):
    '''
    '''
    log.info('Plotting distribution of sky brightness values in image')
    catalog = cfg['Catalog'].get('catalog')
    star_size = int(cfg['Photometry'].getfloat('StarApertureRadius'))
    stars = DM.stars.get(catalog, [])

    plt.rcParams.update({'font.size': 5})
    plt.figure(figsize=(8,6), dpi=100)
    plot_colors = {'R': 'red', 'G': 'green', 'B': 'blue'}
    for c,color in enumerate(DM.sky_brightness.keys()):
        photometry = stars[stars[f'{color}Photometry']]
        good_photometry = stars[stars[f'{color}Photometry'] & ~stars[f'{color}Outliers']]
        outliers = stars[stars[f'{color}Outliers']]

        plt.subplot(3,2,2*c+1)
        binsize = 0.025
        min_bin = np.round(DM.sky_brightness[color]/binsize)*binsize - 10*np.round(DM.sky_brightness_stddev[color]/binsize)*binsize
        max_bin = np.round(DM.sky_brightness[color]/binsize)*binsize + 10*np.round(DM.sky_brightness_stddev[color]/binsize)*binsize
        bins = np.arange(min_bin, max_bin+binsize, binsize)
        plt.hist(good_photometry[f'{color}SkyMag'], bins=bins,
                 color=plot_colors[color], alpha=0.8,
                 label=f'{color} Sky Brightness Values')
        plt.axvline(DM.sky_brightness[color], color='k',
                    label=f'{color} Sky Brightness = {DM.sky_brightness[color]:.2f}')
        plt.axvline(DM.sky_brightness[color]-DM.sky_brightness_stddev[color],
                    color='k', linestyle=':')
        plt.axvline(DM.sky_brightness[color]+DM.sky_brightness_stddev[color],
                    color='k', linestyle=':')
        plt.legend(loc='best')
        plt.ylabel('Number of Stars')
        if c == 2: plt.xlabel('Sky Brightness (mag)')

        plt.subplot(3,2,2*c+2)
        wcs = DM.get_wcs()
        area_of_pixel = wcs.proj_plane_pixel_area().to(u.arcsec**2).value
        color_data = np.array(getattr(DM, plot_colors[color]))/DM.exptime
        log.debug(f'median_filter {color}')
        im_medfilt = scipy.ndimage.median_filter(color_data, size=3*star_size)
        log.debug(f'block_reduce {color}')
        block_size = 200
        im_block_reduced = block_reduce(data=im_medfilt, block_size=block_size, func=np.nanmedian)
        log.debug(f'calulate mag/sq_arcsec {color}')
        im_sb = -2.5*np.log10(im_block_reduced/area_of_pixel) + DM.zero_point[color]
        norm = vis.ImageNormalize(im_sb,
                                  interval=vis.AsymmetricPercentileInterval(1, 99),
                                  stretch=vis.LogStretch())
        log.debug(f'imshow {color}')
        plt.imshow(im_sb, cmap=plt.cm.gray, norm=norm)
        log.debug(f'overlay text values {color}')
        for x in np.arange(0,im_sb.shape[1]):
            for y in np.arange(0,im_sb.shape[0]):
                plt.text(x, y, f"{im_sb[y,x]:.2f}", color='g', size=4)
        plt.xlim(0,im_sb.shape[1])
        plt.ylim(0,im_sb.shape[0])
        plt.xticks([])
        plt.yticks([])

    # Save PNG
    ext = Path(DM.raw_file_name).suffix
    plot_file = Path(DM.raw_file_name.replace(ext, '_sb.png'))
    if plot_file.exists(): plot_file.unlink()
    log.info(f"Saving {str(plot_file)}")
    plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
