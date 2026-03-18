import numpy as np
from astropy import stats
from astropy.table import MaskedColumn

from app import log


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def derive_scaling(DM, reference=None, cfg=None):
    log.info(f'Deriving image scaling parameters: {DM.raw_file_name}')
    catalog = cfg['Catalog'].get('catalog')
    stars = DM.stars.get(catalog, [])
    refstars = reference.stars.get(catalog, [])

    N = len(stars)
    for c in ['R', 'G', 'B']:
        if f'{c}MatchedSkyMean' not in stars.keys():
            stars.add_column(MaskedColumn(data=np.zeros(N, dtype=float),
                             name=f'{c}MatchedSkyMean',
                             mask=np.ones(N, dtype=bool)))
        if f'{c}MatchedStarSum' not in stars.keys():
            stars.add_column(MaskedColumn(data=np.zeros(N, dtype=float),
                             name=f'{c}MatchedStarSum',
                             mask=np.ones(N, dtype=bool)))

    for i,star in enumerate(stars):
        refmatch = (refstars['Source'] == star['Source'])
        assert np.sum(refmatch) <= 1
        try:
            refID = list(refmatch).index(True)
            for c in ['R', 'G', 'B']:
                stars[f'{c}MatchedSkyMean'][i] = refstars[refID][f'{c}SkyMean']
                stars[f'{c}MatchedSkyMean'].mask[i] = False
                stars[f'{c}MatchedStarSum'][i] = refstars[refID][f'{c}StarSum']
                stars[f'{c}MatchedStarSum'].mask[i] = False
        except:
            pass

    background_offset = {}
    flux_scaling = {}
    for c in ['R', 'G', 'B']:
        offset = stars[f'{c}SkyMean'] - stars[f'{c}MatchedSkyMean']
        bko_mean, bko_median, bko_stddev = stats.sigma_clipped_stats(offset[~offset.mask])
        scale = stars[f'{c}StarSum'] / stars[f'{c}MatchedStarSum']
        fls_mean, fls_median, fls_stddev = stats.sigma_clipped_stats(scale[~scale.mask])
        DM.reference = reference.raw_file_name
        DM.background_offset[c] = bko_median #(bko_median, bko_stddev)
        DM.flux_scaling[c] = fls_median #(fls_median, fls_stddev)
