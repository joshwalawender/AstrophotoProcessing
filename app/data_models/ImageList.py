import sys
from pathlib import Path
import re
import numpy as np
from astropy.table import Table, Column, MaskedColumn
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
import ccdproc

from app import log
from app.data_models.OSCImage import OSCImage
from app.reduce.bias_subtract import bias_subtract
from app.reduce.reproject import reproject
from app.analyze.run_astrometrydotnet import solve_field
from app.analyze.get_catalog import query_vizier, apply_catalog
from app.analyze.photometry import photometry
from app.analyze.derive_scaling import derive_scaling


class ImageList(list):
    '''A list containing the image data models (OSCImage objects).
    '''
    def __init__(self, iterable, working_dir=None, masters={},
                 objectname=None,
                 imtype=OSCImage, cfg=None,
                 *args, **kwargs):
        imagelist = []
        iterable = sorted(iterable)
        for i,item in enumerate(iterable):
            if isinstance(item, imtype):
                imagelist.append(item)
            else:
                imagelist.append(imtype(item))
        super().__init__(imagelist, *args, **kwargs)
        self.working_dir = working_dir
        self.imtype = imtype
        self.masters = masters
        self.objectname = objectname
        self.summary_file = working_dir / f'{objectname}.txt'
        self.filters = []
        self.cfg = cfg
#         self.sort(key=lambda x: x.raw_file_name)
        self.reference = 0
        self.initialize_image_metadata_table()


    def add_filter(self, filter_string):
        '''Example filter strings:
        
        [Name] [op] [value]
        FWHM < 8.2
        Elongation < 1.10
        '''
        parsed = re.match('(\\w+)\\s([<>=]+)\\s([\\d\.]+)(%?)', filter_string)
        if parsed is None:
            print(f"Failed to parse: {filter_string}")
            return
        name = parsed.group(1)
        op = parsed.group(2)
        value = float(parsed.group(3))
        pct = parsed.group(4)
        self.filters.append((name, op, value, pct))
        if op == '<' and pct == '':
            log.info(f"Applying filter: {name} {op} {value}")
            new = self.results[name] < value
        elif op == '<' and pct == '%':
            thresh = np.percentile(self.results[name], value)
            log.info(f"Applying filter: {name} {op} {value}{pct} ({thresh})")
            new = self.results[name] < thresh
        n_before = np.sum(self.results['Use?'])
        self.results['Use?'] = self.results['Use?'].data & new
        n_after = np.sum(self.results['Use?'])
        deltaN = n_before-n_after
        log.info(f'  Filter removed {deltaN} data points {n_after} now in use')

    def initialize_image_metadata_table(self):
        self.results = Table([Column([i.ccd.header.get('RAWFILE') for i in self], 'RawFile', str),
                              Column(['']*len(self), 'RA', 'S11'),
                              Column(['']*len(self), 'Dec', 'S12'),
                              Column([np.nan]*len(self), 'WCSOffset', float),
                              Column([np.nan]*len(self), 'FWHM', float),
                              Column([np.nan]*len(self), 'Elongation', float),
                              Column([np.nan]*len(self), 'RZeroPoint', float),
                              Column([np.nan]*len(self), 'RZeroPointStdDev', float),
                              Column([np.nan]*len(self), 'RSkyBrightness', float),
                              Column([np.nan]*len(self), 'RSkyBrightnessStdDev', float),
                              Column([np.nan]*len(self), 'RSkyOffset', float),
                              Column([np.nan]*len(self), 'RFluxScaling', float),
                              Column([np.nan]*len(self), 'GZeroPoint', float),
                              Column([np.nan]*len(self), 'GZeroPointStdDev', float),
                              Column([np.nan]*len(self), 'GSkyBrightness', float),
                              Column([np.nan]*len(self), 'GSkyBrightnessStdDev', float),
                              Column([np.nan]*len(self), 'GSkyOffset', float),
                              Column([np.nan]*len(self), 'GFluxScaling', float),
                              Column([np.nan]*len(self), 'BZeroPoint', float),
                              Column([np.nan]*len(self), 'BZeroPointStdDev', float),
                              Column([np.nan]*len(self), 'BSkyBrightness', float),
                              Column([np.nan]*len(self), 'BSkyBrightnessStdDev', float),
                              Column([np.nan]*len(self), 'BSkyOffset', float),
                              Column([np.nan]*len(self), 'BFluxScaling', float),
                              Column([1]*len(self), 'Use?', int),
                              ])


    def process(self, write=False):
        '''Iterate over entries in list and run each through the specified
        process (e.g. bias_subtract). Hold resulting data models in ImageList.
        '''
        # Get initial guess at center coordinate from object name
        reference_catalog = None
        if self.objectname is not None:
            try:
                result = Simbad.query_object(self.objectname)
                center_coord = SkyCoord(result['ra'].value[0], result['dec'].value[0],
                                        unit=(u.deg, u.deg), frame='icrs')
            except:
                center_coord = None
        else:
            center_coord = None
        
        for i,image in enumerate(self):
            imname = image.ccd.header.get('RAWFILE')
            log.info(f'-----------------------------------------------------------')
            log.info(f'Processing file {i+1}/{len(self)}: {imname}')
            bias_subtract(image, master_bias=self.masters['bias'])
            center_coord = solve_field(image, cfg=self.cfg,
                                       center_coord=center_coord,
                                       search_radius=0.5)
            if center_coord is None:
                log.warning(f'Plate solve for {imname} failed')
            else:
                if reference_catalog is None:
                    reference_catalog = query_vizier(image, cfg=self.cfg)
                apply_catalog(image, reference_catalog, cfg=self.cfg)
                photometry(image, cfg=self.cfg)
            # Record image results to table
            radecstr = image.center_coord.to_string('hmsdms', sep=':', precision=2)
            self.results[i]['RA'] = radecstr.split()[0]
            self.results[i]['Dec'] = radecstr.split()[1]
            self.results[i]['WCSOffset'] = image.ccd.header.get('WCSMDOFF')
            self.results[i]['FWHM'] = image.ccd.header.get('FWHM')
            self.results[i]['Elongation'] = image.ccd.header.get('ELONG')
            for c in ['R', 'G', 'B']:
                self.results[i][f'{c}ZeroPoint'] = image.ccd.header.get(f'{c}ZEROPT')
                self.results[i][f'{c}ZeroPointStdDev'] = image.ccd.header.get(f'{c}ZEROSD')
                self.results[i][f'{c}SkyBrightness'] = image.ccd.header.get(f'{c}SKYB')
                self.results[i][f'{c}SkyBrightnessStdDev'] = image.ccd.header.get(f'{c}SKYBSD')


    def set_reference_image(self, optimizer, op='min'):
        assert optimizer in self.results.keys()
        if op == 'min':
            self.reference = np.argmin(self.results[optimizer])
            value = self.results[optimizer][self.reference]
            log.info(f"Minimum {optimizer} is {value} for frame {self.reference}")
        if op == 'max':
            self.reference = np.argmax(self.results[optimizer])
            value = self.results[optimizer][self.reference]
            log.info(f"Maximum {optimizer} is {value} for frame {self.reference}")
        reference_image = self[self.reference]
        
        reference_filename = reference_image.ccd.header.get('RAWFILE').replace('.fit', '_processed.fits')
        log.info(f'-----------------------------------------------------------')
        log.info(f'Determining Scaling: Reference is {reference_filename}')
        for i,image in enumerate(self):
            derive_scaling(image, reference=reference_image, cfg=self.cfg)
            for c in ['R', 'G', 'B']:
                self.results[f'{c}FluxScaling'] = image.ccd.header.get(f'{c}SCALE')
                self.results[f'{c}SkyOffset'] = image.ccd.header.get(f'{c}BKGOFF')


    def reproject(self):
        '''Re-project all images on to WCS of the reference image
        '''
        reference_image = self[self.reference]
        reference_filename = reference_image.ccd.header.get('RAWFILE').replace('.fit', '_processed.fits')
        log.info(f'-----------------------------------------------------------')
        log.info(f'Reprojecting Images: Reference is {reference_filename}')
        reference_wcs = reference_image.ccd.wcs
        for i,image in enumerate(self):
            filename = image.ccd.header.get('RAWFILE')
            log.info(f'Reprojecting file {i+1}/{len(self)}: {filename}')
            reproject(image, reference_wcs=reference_wcs)


    def combine(self):
        log.info(f'-----------------------------------------------------------')
        log.info(f'Combining Images:')
        reds = []
        red_scale = []
        greens = []
        green_scale = []
        blues = []
        blue_scale = []
        for i,image in enumerate(self):
            filename = image.ccd.header.get('RAWFILE')
            if self.results[i]['Use?'] == True:
                log.info(f'Opening file {i+1}/{len(self)} for stacking: {filename}')
                reds.append(image.red)
                red_scale.append(self.results['RFluxScaling'][i])
                greens.append(image.green)
                green_scale.append(self.results['GFluxScaling'][i])
                blues.append(image.blue)
                blue_scale.append(self.results['BFluxScaling'][i])

        log.info(f'Combining {len(reds)} red images')
        red_combiner = ccdproc.Combiner(reds)
#         red_combiner.scaling=red_scale
#         red_stacked = red_combiner.average_combine()
#         red_stacked = red_combiner.clip_extrema(nhigh=3, nlow=1)
        red_stacked = red_combiner.sigma_clipping(low_thresh=5, high_thresh=5, func=np.ma.median)
        red_file = Path('red.fits')
        if red_file.exists(): red_file.unlink()
        ccdproc.fits_ccddata_writer(red_stacked, red_file)

        log.info(f'Combining {len(greens)} green images')
        green_combiner = ccdproc.Combiner(greens)
#         green_combiner.scaling=green_scale
#         green_stacked = green_combiner.average_combine()
#         green_stacked = green_combiner.clip_extrema(nhigh=3, nlow=1)
        green_stacked = green_combiner.sigma_clipping(low_thresh=5, high_thresh=5, func=np.ma.median)
        green_file = Path('green.fits')
        if green_file.exists(): green_file.unlink()
        ccdproc.fits_ccddata_writer(green_stacked, green_file)

        log.info(f'Combining {len(blues)} blue images')
        blue_combiner = ccdproc.Combiner(blues)
#         blue_combiner.scaling=blue_scale
#         blue_stacked = blue_combiner.average_combine()
#         blue_stacked = blue_combiner.clip_extrema(nhigh=3, nlow=1)
        blue_stacked = blue_combiner.sigma_clipping(low_thresh=5, high_thresh=5, func=np.ma.median)
        blue_file = Path('blue.fits')
        if blue_file.exists(): blue_file.unlink()
        ccdproc.fits_ccddata_writer(blue_stacked, blue_file)
        
    def write_all(self):
        for image in self:
			raw_file_name = image.ccd.header.get('RAWFILE')
            working_file_name = raw_file_name.replace('.fit', '.fits')
            working_file = self.working_dir / working_file_name
            log.info(f'Writing {working_file.name}')
            image.write(working_file)


    def plot_image_quality(self):
        use = np.array(self.results['Use?'].data, dtype=bool)
        imcount = np.arange(0,len(self),1,dtype=int)

        plt.figure(figsize=(6,6))
        plt.rcParams.update({'font.size': 4})

        plt.subplot(3,1,1)
        plt.title(f'FWHM Over Time')
        plt.plot(imcount, self.results['FWHM'], 'ko-')
        plt.plot(imcount[~use], self.results['FWHM'][~use], 'rx')
        plt.plot(imcount[self.reference], self.results['FWHM'][self.reference], 'g+')
        plt.ylabel('FWHM (pix)')
        plt.grid()

        plt.subplot(3,1,2)
        plt.title(f'Elongation Over Time')
        plt.plot(imcount, self.results['Elongation'], 'ko-')
        plt.plot(imcount[~use], self.results['Elongation'][~use], 'rx')
        plt.plot(imcount[self.reference], self.results['Elongation'][self.reference], 'g+')
        plt.xlabel('Sequence Number')
        plt.ylabel('Elongation')
        plt.grid()

        plt.subplot(3,1,3)
        plt.title(f'WCSOffset Over Time')
        plt.plot(imcount, self.results['WCSOffset'], 'ko-')
        plt.plot(imcount[~use], self.results['WCSOffset'][~use], 'rx')
        plt.plot(imcount[self.reference], self.results['WCSOffset'][self.reference], 'g+')
        plt.xlabel('Sequence Number')
        plt.ylabel('WCS Offset')
        plt.grid()

        # Save PNG
        ext = Path(self.summary_file).suffix
        plot_file = Path(self.summary_file.name.replace(ext, '_IQ.png'))
        if plot_file.exists(): plot_file.unlink()
        log.info(f"Saving {str(plot_file)}")
        plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.1, dpi=200)


    def plot_photometry(self):
        use = np.array(self.results['Use?'].data, dtype=bool)
        imcount = np.arange(0,len(self),1,dtype=int)

        plt.figure(figsize=(6,4))
        plt.rcParams.update({'font.size': 4})

        plt.subplot(2,1,1)
        plt.title(f'Zero Point Over Time')
        for c in ['R', 'G', 'B']:
            plt.plot(imcount, self.results[f'{c}ZeroPoint'], f'{c.lower()}o-')
            plt.plot(imcount[~use], self.results[f'{c}ZeroPoint'][~use], 'kx')
        plt.ylabel('Zero Point (mag)')
        plt.grid()

        plt.subplot(2,1,2)
        plt.title(f'Sky Brightness Over Time')
        for c in ['R', 'G', 'B']:
            plt.plot(imcount, self.results[f'{c}SkyBrightness'], f'{c.lower()}o-')
            plt.plot(imcount[~use], self.results[f'{c}SkyBrightness'][~use], 'kx')
        plt.ylabel('Sky Brightness (mag/arcsec^2)')
        plt.grid()

        # Save PNG
        ext = Path(self.summary_file).suffix
        plot_file = Path(self.summary_file.name.replace(ext, '_Photometry.png'))
        if plot_file.exists(): plot_file.unlink()
        log.info(f"Saving {str(plot_file)}")
        plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.1, dpi=200)
