#!python3

## Import General Tools
from pathlib import Path
from matplotlib import pyplot as plt


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def plot_zeropoints(datamodel, cfg=None):
    '''
    '''
    catalog = cfg['Catalog'].get('catalog')
    stars = datamodel.stars.get(catalog, [])

    plt.figure(figsize=(12,6))
    plt.hist(stars['ZeroPoint'][stars['Photometry'] == True], bins=100,
             label='Zero Point Values')
    plt.axvline(datamodel.zero_point, color='k',
                label=f'Zero Point = {datamodel.zero_point:.3f}')
    plt.axvline(datamodel.zero_point-datamodel.zero_point_stddev,
                color='k', linestyle=':')
    plt.axvline(datamodel.zero_point+datamodel.zero_point_stddev, color='k', linestyle=':')
    plt.legend(loc='best')

    # Save PNG
    ext = Path(datamodel.raw_file_name).suffix
    plot_file = Path(datamodel.raw_file_name.replace(ext, '_zp.png'))
    if plot_file.exists(): plot_file.unlink()
    plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
