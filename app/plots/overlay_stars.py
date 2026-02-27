#!python3

## Import General Tools
from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy import visualization as vis
from astropy.coordinates import SkyCoord, ICRS

from matplotlib import pyplot as plt


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def overlay_stars(datamodel, cfg=None):
    '''Take the catalog stars and WCS and generate a PNG file 
    '''
    catalog = cfg['Catalog'].get('catalog')
    star_r = cfg['Photometry'].getfloat('StarApertureRadius')
    stars = datamodel.stars.get(catalog, [])
    
    norm = vis.ImageNormalize(datamodel.data,
                              interval=vis.AsymmetricPercentileInterval(1, 99.99),
                              stretch=vis.LogStretch())
    plt.figure(figsize=(12,12), dpi=300)
    plt.imshow(datamodel.data, cmap=plt.cm.gray, norm=norm)
    plt.xlim(0,datamodel.data.shape[1])
    plt.ylim(0,datamodel.data.shape[0])
    plt.xticks([])
    plt.yticks([])

    # Overlay Catalog Star Positions as WCS Evaluation
    print(f"Overlaying catalog star positions")
    plt.scatter(stars['Catalog_X'], stars['Catalog_Y'],
                s=3*star_r, c='y', marker='+',
                linewidths=0.5, edgecolors=None, alpha=0.5)

    # Overlay Centroided Star Positions
#     centroided = ~np.isnan(stars['Centroid_X'])
#     print(f"Overlaying {np.sum(centroided)} centroided star positions")
#     for star in stars[centroided]:
#         c = plt.Circle((star['Centroid_X'], star['Centroid_Y']),
#                        radius=2*star_r, edgecolor='b', facecolor='none',
#                        alpha=0.5)
#         plt.gca().add_artist(c)

    # Overlay Stars with Good Photometry
    good_photometry = stars[stars['Photometry'] & ~stars['Outliers']]
    print(f"Overlaying {len(good_photometry)} stars with good photometry")
    for star in good_photometry:
        c = plt.Circle((star['Centroid_X'], star['Centroid_Y']),
                       radius=star_r, edgecolor='b', facecolor='none',
                       alpha=0.5)
        plt.gca().add_artist(c)
        c = plt.Circle((star['Centroid_X'], star['Centroid_Y']),
                       radius=2*star_r, edgecolor='b', facecolor='none',
                       alpha=0.5)
        plt.gca().add_artist(c)
        c = plt.Circle((star['Centroid_X'], star['Centroid_Y']),
                       radius=3*star_r, edgecolor='b', facecolor='none',
                       alpha=0.5)
        plt.gca().add_artist(c)

    # Overlay Photometry Outliers
    outliers = stars[stars['Outliers']]
    print(f"Overlaying {len(outliers)} photometry outlier stars")
    for star in outliers:
        c = plt.Circle((star['Centroid_X'], star['Centroid_Y']),
                       radius=star_r, edgecolor='r', facecolor='none',
                       alpha=0.5)
        plt.gca().add_artist(c)
        c = plt.Circle((star['Centroid_X'], star['Centroid_Y']),
                       radius=2*star_r, edgecolor='r', facecolor='none',
                       alpha=0.5)
        plt.gca().add_artist(c)
        c = plt.Circle((star['Centroid_X'], star['Centroid_Y']),
                       radius=3*star_r, edgecolor='r', facecolor='none',
                       alpha=0.5)
        plt.gca().add_artist(c)

    # Save JPEG
    ext = Path(datamodel.raw_file_name).suffix
    jpeg_file = Path(datamodel.raw_file_name.replace(ext, '.jpg'))
    if jpeg_file.exists(): jpeg_file.unlink()
    print(f"Saving {str(jpeg_file)}")
    plt.savefig(jpeg_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
