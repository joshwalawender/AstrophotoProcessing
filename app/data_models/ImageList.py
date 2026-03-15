from pathlib import Path
import re
import numpy as np
from astropy.table import Table, Column, MaskedColumn
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt

from app import log
from app.data_models.OSCImage import OSCImage
from app.reduce.bias_subtract import bias_subtract
from app.analyze.run_astrometrydotnet import solve_field
from app.analyze.get_catalog import query_vizier, apply_catalog
from app.analyze.photometry import photometry
from app.analyze.derive_scaling import derive_scaling


class ImageList(list):
    '''The main list contain the image data models. This may be bad use of memory?

    Hold master files in dictionary attached to this ImageList.

    Add method to iterate over contents, read properties and output table of
    image properties?

    Method to iterate over list, run the specified functions and optionally
    save out files?
    '''
    def __init__(self, iterable, working_dir=None, masters={},
                 objectname=None,
                 imtype=OSCImage, cfg=None,
                 *args, **kwargs):
        super().__init__(iterable, *args, **kwargs)
        self.working_dir = working_dir
        self.imtype = imtype
        self.masters = masters
        self.objectname = objectname
        self.summary_file = Path(f'{objectname}_results.txt')
        self.filters = []
        self.cfg = cfg
        self.sort()
        self.reference = 0
        self.initialize_image_metadata_table()


    def add_filter(self, filter_string):
        '''Example filter strings:
        
        [Name] [op] [value]
        FWHM < 8.2
        Elongation < 1.10
        '''
        parsed = re.match('(\w+)\s([<>=]+)\s([\d\.]+)(%?)', filter_string)
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
        self.results = Table([Column([rf for rf in self], 'RawFile', str),
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
        for i in range(len(self)):
            working_file = self.working_dir / self[i].name.replace('.fit', '_processed.fits')
            log.info(f'-----------------------------------------------------------')
            log.info(f'Processing file {i+1}/{len(self)}: {working_file.name}')
            if i==0:
                image = self.imtype(self[i])
                bias_subtract(image, master_bias=self.masters['bias'])

                if self.objectname is not None:
                    try:
                        result = Simbad.query_object(self.objectname)
                        estimated_center = SkyCoord(result['ra'].value[0], result['dec'].value[0],
                                                    unit=(u.deg, u.deg), frame='icrs')
                    except:
                        estimated_center = None
                else:
                    estimated_center = None
                center_coord = solve_field(image, cfg=self.cfg,
                                           center_coord=estimated_center,
                                           search_radius=1)
                reference_catalog = query_vizier(image, cfg=self.cfg)
                photometry(image, cfg=self.cfg)
                log.info(f'Writing processed file: {working_file.name}')
                image.write(working_file)
            else:
                if working_file.exists():
                    log.info(f'Found existing file: {working_file.name}')
                    image = self.imtype(working_file)
                else:
                    image = self.imtype(self[i])
                    bias_subtract(image, master_bias=self.masters['bias'])
                    success = solve_field(image, cfg=self.cfg,
                                          center_coord=center_coord)
                    if success is None:
                        log.warning(f'Image {working_file.name} failed')
                    else:
                        apply_catalog(image, reference_catalog, cfg=self.cfg)
                        photometry(image, cfg=self.cfg)
                        log.info(f'Writing processed file: {working_file.name}')
                        image.write(working_file)
            # Record image results to table
            radecstr = image.center_coord.to_string('hmsdms', sep=':', precision=2)
            self.results[i]['RA'] = radecstr.split()[0]
            self.results[i]['Dec'] = radecstr.split()[1]
            self.results[i]['WCSOffset'] = image.WCS_median_offset
            self.results[i]['FWHM'] = image.fwhm
            self.results[i]['Elongation'] = image.elongation
            self.results[i]['RZeroPoint'] = image.zero_point['R']
            self.results[i]['RZeroPointStdDev'] = image.zero_point_stddev['R']
            self.results[i]['RSkyBrightness'] = image.sky_brightness['R']
            self.results[i]['RSkyBrightnessStdDev'] = image.sky_brightness_stddev['R']
            self.results[i]['GZeroPoint'] = image.zero_point['G']
            self.results[i]['GZeroPointStdDev'] = image.zero_point_stddev['G']
            self.results[i]['GSkyBrightness'] = image.sky_brightness['G']
            self.results[i]['GSkyBrightnessStdDev'] = image.sky_brightness_stddev['G']
            self.results[i]['BZeroPoint'] = image.zero_point['B']
            self.results[i]['BZeroPointStdDev'] = image.zero_point_stddev['B']
            self.results[i]['BSkyBrightness'] = image.sky_brightness['B']
            self.results[i]['BSkyBrightnessStdDev'] = image.sky_brightness_stddev['B']


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


    def reproject(self):
        '''Re-project all images on to WCS of the reference image
        '''
        reference_filename = self[self.reference].name.replace('.fit', '_processed.fits')
        reference_file = self.working_dir / reference_filename
        log.info(f'-----------------------------------------------------------')
        log.info(f'Reprojecting Images: Reference is {reference_filename}')
        reference_image = self.imtype(reference_file)
        reference_wcs = reference_image.get_wcs()
        for i in range(len(self)):
            working_file = self.working_dir / self[i].name.replace('.fit', '_processed.fits')
            log.info(f'Reprojecting file {i+1}/{len(self)}: {working_file.name}')
            image = self.imtype(working_file)
            reproject(image, reference_wcs=reference_wcs)


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
