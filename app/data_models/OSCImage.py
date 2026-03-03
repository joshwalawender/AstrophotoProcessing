from pathlib import Path
import sys
import copy
import tempfile
import datetime

import numpy as np
from astropy import units as u
from astropy.io import fits

from astropy.io.fits.verify import VerifyWarning
import warnings
warnings.simplefilter('ignore', category=VerifyWarning)

from astropy.nddata import CCDData, block_reduce
from astropy.table import Table
from astropy.coordinates import SkyCoord, ICRS
from astropy import wcs
from astropy import visualization as vis
from astropy import samp
import ccdproc
# from regions import Regions, PixCoord, CirclePixelRegion

from matplotlib import pyplot as plt

from app import log


class OSCImage(object):
    def __init__(self, hdulist, *args, **kwargs):
        if isinstance(hdulist, str):
            hdulist = Path(hdulist).expanduser().absolute()
        if isinstance(hdulist, Path):
            hdulist = hdulist.expanduser().absolute()
            assert hdulist.exists()
            self.raw_file_name = hdulist.name
            log.debug(f'Instantiating data model from {self.raw_file_name}')
            hdulist = fits.open(hdulist)
        else:
            print(f"HDUList input {type(hdulist)} is unknown")
            sys.exit(1)
        self.hdulist = hdulist
        self.hdulist.verify("fix")
        self.hdu_names = [hdu.name for hdu in self.hdulist]
        self.WCS_median_offset = None
        self.fwhm = None
        self.fwhm_stddev = None
        self.elongation = None
        self.elongation_stddev = None
        self.zero_point = {}
        self.zero_point_stddev = {}
        self.sky_brightness = {}
        self.sky_brightness_stddev = {}
        self.reference = None
        self.background_offset = {}
        self.flux_scaling = {}
        self.tempdir = Path(tempfile.mkdtemp())

        # Connect to ds9 via SAMP
        try:
            self.ds9 = samp.SAMPIntegratedClient()
            self.ds9.connect()
        except samp.errors.SAMPHubError as e:
            self.ds9 = None
        except Exception as e:
            print('Unable to connect to ds9')
            print(type(e))
            print(e)
            self.ds9 = None

        # Assume first extension is primary (Raw data)
        assert self.hdu_names[0] == 'PRIMARY'
        self.exptime = float(self.hdulist[0].header.get('EXPTIME', 0))

        # Create PROCESSED extension if needed
        if 'PROCESSED' not in self.hdu_names:
            # This is a new data model
            # Build PROCESSED Extension
            header = fits.Header()
            header.set('APP_DM', 'OSCImage', 'Which type of APP data model is this?')
            processed_hdu = fits.ImageHDU(data=hdulist[0].data,
                                          header=header,
                                          name='PROCESSED')
            self.hdulist.append(processed_hdu)
            self.data = CCDData(data=copy.deepcopy(processed_hdu.data),
                                meta={'APP_DM': 'OSCImage'},
                                unit='adu',
                                )
            self.build_color_mask()
            self.center_coord = None
            self.radius = None
        else:
            processed = self.getHDU('PROCESSED')
            # Check this is our data model
            assert self.hdulist[processed].header.get('APP_DM') == 'OSCImage'
            self.data = CCDData(data=self.hdulist[processed].data,
                                meta={'APP_DM': 'OSCImage'},
                                unit='adu',
                                )
            self.build_color_mask()
            ra_str = self.hdulist[processed].header.get('CENT_RA', None)
            dec_str = self.hdulist[processed].header.get('CENT_DEC', None)
            if ra_str and dec_str:
                self.center_coord = SkyCoord(ra_str, dec_str, frame=ICRS,
                                             unit=(u.hourangle, u.deg))
            else:
                self.center_coord = None
            fovrad = self.hdulist[processed].header.get('FOVRAD', None)
            if fovrad:
                self.radius = float(fovrad)
            else:
                self.radius = None
            self.raw_file_name = self.hdulist[processed].header.get('RAWNAME', None)

            
            # Read in Individual Color Images
#             for color in ['Red', 'Green', 'Blue']:
#                 ind = self.getHDU(color)
#                 data = CCDData(data=self.hdulist[ind].data,
#                                meta={'APP_DM': 'OSCImage', 'COLOR': color},
#                                unit='adu',
#                                )
#                 setattr(self, color.lower(), data)

            # Read in FWHM Values
            hdr_fwhm = self.hdulist[processed].header.get('FWHM', None)
            self.fwhm = float(hdr_fwhm) if hdr_fwhm is not None else None
            hdr_fwhm_std = self.hdulist[processed].header.get('FWHMSTD', None)
            self.fwhm_stddev = float(hdr_fwhm_std) if hdr_fwhm_std is not None else None
            # Read in elongation Values
            hdr_elng = self.hdulist[processed].header.get('ELNG', None)
            self.elongation = float(hdr_elng) if hdr_elng is not None else None
            hdr_elng_std = self.hdulist[processed].header.get('ELNGSTD', None)
            self.elongation_stddev = float(hdr_elng_std) if hdr_elng_std is not None else None

            # Read in WCS Offset Value
            hdr_WCS_median_offset = self.hdulist[processed].header.get('WCSOFFST', None)
            self.WCS_median_offset = float(hdr_WCS_median_offset) if hdr_WCS_median_offset is not None else None

            # Read in Photometry Reference Comparison Values
            hdr_ref = self.hdulist[processed].header.get('PHOTREF', None)
            self.reference = hdr_ref if hdr_ref is not None else None

            # Read in Zero Point Values
            for color in ['R', 'G', 'B']:
                hdr_zp = self.hdulist[processed].header.get(f'{color}ZEROPNT', None)
                if hdr_zp: self.zero_point[color] = float(hdr_zp)
                hdr_zp_std = self.hdulist[processed].header.get(f'{color}ZPSTD', None)
                if hdr_zp_std: self.zero_point_stddev[color] = float(hdr_zp_std)
                hdr_sb = self.hdulist[processed].header.get(f'{color}MSKY', None)
                if hdr_sb: self.sky_brightness[color] = float(hdr_sb)
                hdr_sb_std = self.hdulist[processed].header.get(f'{color}MSKYSTD', None)
                if hdr_sb_std: self.sky_brightness_stddev[color] = float(hdr_sb_std)
                hdr_bo = self.hdulist[processed].header.get(f'{color}SKYOFF', None)
                if hdr_bo: self.background_offset[color] = float(hdr_bo)
                hdr_fs = self.hdulist[processed].header.get(f'{color}FLUXSCL', None)
                if hdr_fs: self.flux_scaling[color] = float(hdr_fs)


        self.split_colors()

        # Build Catalogs
        if 'Gaia DR3' not in self.hdu_names:
            self.stars = {} # Dictionary of tables for catalog stars
        else:
            gaiaDR3 = self.getHDU('Gaia DR3')
            self.stars = {'Gaia DR3': Table(self.hdulist[gaiaDR3].data)}


    ##-------------------------------------------------------------------------
    ## getHDU
    ##-------------------------------------------------------------------------
    def getHDU(self, name):
        self.hdu_names = [hdu.name for hdu in self.hdulist]
        try:
            ind = self.hdu_names.index(name)
        except ValueError:
            ind = -1
        return ind


    ##-------------------------------------------------------------------------
    ## build_color_mask
    ##-------------------------------------------------------------------------
    def build_color_mask(self):
        self.bayer = []
        for row in range(self.data.shape[0]):
            if row%2==0: self.bayer.append(['R', 'G']*int(self.data.shape[1]/2))
            if row%2!=0: self.bayer.append(['G', 'B']*int(self.data.shape[1]/2))
        self.bayer = np.array(self.bayer)
        self.mask = {'R': self.bayer == 'R',
                     'G': self.bayer == 'G',
                     'B': self.bayer == 'B'}

    ##-------------------------------------------------------------------------
    ## split_colors
    ##-------------------------------------------------------------------------
    def split_colors(self):
        red = block_reduce(np.ma.MaskedArray(data=self.data, mask=self.mask['R']),
                           2, func=np.nanmean)
        self.red = CCDData(data=red, unit='adu',
                           meta={'APP_DM': 'OSCImage', 'COLOR': 'Red'})
        green = block_reduce(np.ma.MaskedArray(data=self.data, mask=self.mask['G']),
                           2, func=np.nanmean)
        self.green = CCDData(data=green, unit='adu',
                           meta={'APP_DM': 'OSCImage', 'COLOR': 'Green'})
        blue = block_reduce(np.ma.MaskedArray(data=self.data, mask=self.mask['B']),
                           2, func=np.nanmean)
        self.blue = CCDData(data=blue, unit='adu',
                           meta={'APP_DM': 'OSCImage', 'COLOR': 'Blue'})

    ##-------------------------------------------------------------------------
    ## update_data
    ##-------------------------------------------------------------------------
    def update_data(self, newdata, header=[], history=[]):
        '''Update the .data attribute (the CCDData object) and add any header
        and history lines to the .hdulist[1].header
        '''
        # Image Data
        if isinstance(newdata, CCDData):
            self.data = newdata
            self.split_colors()
        # Header
        pind = self.getHDU('PROCESSED')
        for h in header:
            self.hdulist[pind].header.set(*h)
        # History
        now = datetime.datetime.now()
        nowstr = now.strftime('%Y-%m-%D %H:%M:%S')
        for historyline in history:
            self.hdulist[1].header.add_history(f"{nowstr}: {historyline}")


    ##-------------------------------------------------------------------------
    ## get_wcs
    ##-------------------------------------------------------------------------
    def get_wcs(self):
        processed = self.getHDU('PROCESSED')
        if processed >= 0:
            return wcs.WCS(self.hdulist[processed].header)
        else:
            return None


    ##-------------------------------------------------------------------------
    ## write_tmp
    ##-------------------------------------------------------------------------
    def write_tmp(self):
        '''Write a single extension FITS file for temporary use (e.g. by
        astrometry.net)
        '''
        tfile = self.tempdir / 'app_temp_image.fits'
        ccdproc.fits_ccddata_writer(self.data, tfile)
        return tfile


    ##-------------------------------------------------------------------------
    ## write
    ##-------------------------------------------------------------------------
    def write(self, filename, overwrite=True):
        '''Write a Multi-Extension FITS file to hold the entire data model.
        '''
        assert len(self.data.to_hdu(as_image_hdu=True)) == 1
        # Processed Image Data
        pind = self.getHDU('PROCESSED')
        phdu = self.data.to_hdu(as_image_hdu=True)[0]
        self.hdulist[pind].data = phdu.data
        # Header info
        cra, cdec = self.center_coord.to_string('hmsdms', sep=':', precision=2).split()
        self.hdulist[pind].header.set('CENT_RA', cra,
                                      'RA coordinate at center of FoV')
        self.hdulist[pind].header.set('CENT_DEC', cdec,
                                      'Dec coordinate at center of FoV')
        self.hdulist[pind].header.set('FOVRAD', f'{self.radius:.3f}',
                                      'Radius encompassing FoV [degrees]')
        self.hdulist[pind].header.set('RAWNAME', str(self.raw_file_name),
                                      'Original (raw) file name')


        if self.WCS_median_offset:
            self.hdulist[pind].header.set('WCSOFFST', f'{self.WCS_median_offset:.2f}',
                                          'Median Offset from WCS [pix]')
        if self.fwhm:
            self.hdulist[pind].header.set('FWHM', f'{self.fwhm:.2f}',
                                          'Typical FWHM')
            self.hdulist[pind].header.set('FWHMSTD', f'{self.fwhm_stddev:.2f}',
                                          'FWHM Std Dev')
        if self.elongation:
            self.hdulist[pind].header.set('ELNG', f'{self.elongation:.2f}',
                                          'Typical elongation')
            self.hdulist[pind].header.set('ELNGSTD', f'{self.elongation_stddev:.2f}',
                                          'elongation Std Dev')
        for color in self.zero_point.keys():
            self.hdulist[pind].header.set(f'{color}ZEROPNT', f'{self.zero_point.get(color):.3f}',
                                          f'Calculated Zero Point ({color})')
            self.hdulist[pind].header.set(f'{color}ZPSTD', f'{self.zero_point_stddev.get(color):.3f}',
                                          f'Calculated Zero Point Std Dev ({color})')
        for color in self.sky_brightness.keys():
            self.hdulist[pind].header.set(f'{color}MSKY', f'{self.sky_brightness.get(color):.3f}',
                                          f'Sky Brigtness ({color}) [mag/arcsec^2]')
            self.hdulist[pind].header.set(f'{color}MSKYSTD', f'{self.sky_brightness_stddev.get(color):.3f}',
                                          f'Sky Brigtness Std Dev ({color}) [mag/arcsec^2]')
        if self.reference:
            self.hdulist[pind].header.set('PHOTREF', self.reference,
                                          'Reference for photometry comparison (cSKYOFF, cFLUXSCL)')
        for color in self.background_offset.keys():
            self.hdulist[pind].header.set(f'{color}SKYOFF', f'{self.background_offset.get(color):.3f}',
                                          f'Sky Brigtness Offset ({color}) [ADU]]')
            self.hdulist[pind].header.set(f'{color}FLUXSCL', f'{self.flux_scaling.get(color):.3f}',
                                          f'Flux Scaling Factor ({color})')

        # Three Colors
#         for color in ['Red', 'Green', 'Blue']:
#             cind = self.getHDU(color)
#             chdu = getattr(self, color.lower()).to_hdu(as_image_hdu=True)[0]
#             chdu.header.set('EXTNAME', color)
#             if cind == -1:
#                 self.hdulist.append(chdu)
#             else:
#                 self.hdulist[cind] = chdu

        # Write Catalog Stars to FITS Table
        for catalog in self.stars.keys():
            FITStable = fits.table_to_hdu(self.stars[catalog])
            FITStable.header.set('EXTNAME', catalog)
            HDUind = self.getHDU(catalog)
            if HDUind >= 0:
                self.hdulist[HDUind] = FITStable
            else:
                self.hdulist.append(FITStable)
        # Write to file
        self.hdulist.writeto(filename, overwrite=overwrite)


    ##-------------------------------------------------------------------------
    ## write_jpg
    ##-------------------------------------------------------------------------
    def write_jpg(self, output=None, radius=8):
        '''Take the catalog stars and WCS and generate a PNG file 
        '''
        log.info('Generating JPG image')
        catalog = list(self.stars.keys())[0]
        stars = self.stars.get(catalog, [])

        norm = vis.ImageNormalize(self.data,
                                  interval=vis.AsymmetricPercentileInterval(1, 99.99),
                                  stretch=vis.LogStretch())
        plt.figure(figsize=(12,12), dpi=300)
        plt.imshow(self.data, cmap=plt.cm.gray, norm=norm)
        plt.xlim(0,self.data.shape[1])
        plt.ylim(0,self.data.shape[0])
        plt.xticks([])
        plt.yticks([])

        # Overlay Catalog Star Positions as WCS Evaluation
        log.info(f"  Overlaying catalog star positions")
        plt.scatter(stars['Catalog_X'], stars['Catalog_Y'],
                    s=3*radius, c='y', marker='+',
                    linewidths=0.5, edgecolors=None, alpha=0.5)

        # Overlay Stars with Good G Photometry
        good_photometry = stars[stars['GPhotometry'] & ~stars['GOutliers']]
        log.info(f"  Overlaying {len(good_photometry)} stars with good G photometry")
        for star in good_photometry:
            c = plt.Circle((star['Centroid_X'], star['Centroid_Y']),
                           radius=radius, edgecolor='b', facecolor='none',
                           alpha=0.5)
            plt.gca().add_artist(c)
            c = plt.Circle((star['Centroid_X'], star['Centroid_Y']),
                           radius=2*radius, edgecolor='b', facecolor='none',
                           alpha=0.5)
            plt.gca().add_artist(c)
            c = plt.Circle((star['Centroid_X'], star['Centroid_Y']),
                           radius=3*radius, edgecolor='b', facecolor='none',
                           alpha=0.5)
            plt.gca().add_artist(c)

        # Overlay Photometry Outliers
        outliers = stars[stars['GOutliers']]
        log.info(f"  Overlaying {len(outliers)} G photometry outlier stars")
        for star in outliers:
            c = plt.Circle((star['Centroid_X'], star['Centroid_Y']),
                           radius=radius, edgecolor='r', facecolor='none',
                           alpha=0.5)
            plt.gca().add_artist(c)
            c = plt.Circle((star['Centroid_X'], star['Centroid_Y']),
                           radius=2*radius, edgecolor='r', facecolor='none',
                           alpha=0.5)
            plt.gca().add_artist(c)
            c = plt.Circle((star['Centroid_X'], star['Centroid_Y']),
                           radius=3*radius, edgecolor='r', facecolor='none',
                           alpha=0.5)
            plt.gca().add_artist(c)

        # Save JPEG
        rawext = Path(self.raw_file_name).suffix
        if output is None:
            jpeg_file = Path(self.raw_file_name.replace(rawext, '.jpg'))
        elif Path(output).is_dir():
            jpeg_file = Path(output).expanduser() / self.raw_file_name.replace(rawext, '.jpg')
        else:
            jpeg_file = Path(output).expanduser()
        if jpeg_file.exists(): jpeg_file.unlink()
        log.info(f"Saving {str(jpeg_file)}")
        plt.savefig(jpeg_file, bbox_inches='tight', pad_inches=0.1, dpi=300)


    ##-------------------------------------------------------------------------
    ## display
    ##-------------------------------------------------------------------------
    def ds9_set(self, cmd):
        if self.ds9:
            msg = {"samp.mtype": "ds9.set",
                   "samp.params":{"cmd": cmd}}
            self.ds9.notify_all(msg)


    def ds9_get(self, cmd):
        if self.ds9:
            msg = {"samp.mtype": "ds9.get",
                   "samp.params":{"cmd": cmd}}
            self.ds9.notify_all(msg)


#     def display_using_set(self):
#         if self.ds9:
#             tempfile = self.tempdir / 'app_temp_image.fits'
#             x, y = self.data.shape
#             fp = np.memmap(tempfile, dtype='float32', mode='w+', shape=self.data.shape)
#             fp[:] = np.array(self.data)[:]
#             fp.flush()
#             self.ds9_set(f"array {str(tempfile)}[xdim={x:d},ydim={y:d},bitpix=-32]")
#             tempfile.unlink()


    def display(self):
        if self.ds9:
            tempfile = self.tempdir / 'app_temp_image.fits'
            x, y = self.data.shape
            fp = np.memmap(tempfile, dtype='float32', mode='w+', shape=self.data.shape)
            fp[:] = np.array(self.data)[:]
            fp.flush()
            cmd = f"array {str(tempfile)}[xdim={x:d},ydim={y:d},bitpix=-32]"
            self.ds9.ecall_and_wait("c1", "ds9.set", "10", cmd=cmd)
            tempfile.unlink()


#     def regions_from_catalog(self, catalog='Gaia DR3', radius=10):
#         if self.ds9:
#             xys = [PixCoord(x=star['Catalog_X'], y=star['Catalog_Y']) for star in self.stars.get(catalog, [])]
#             reglist = Regions([CirclePixelRegion(xy, radius) for xy in xys])
#             reglist.write('tmp.reg')
#             cmd = f'region load /Users/jwalawender/git/AstrophotoProcessing/tmp.reg'
#             self.ds9_set(cmd)
#         for star in self.stars.get(catalog, []):
#             xy = PixCoord(x=star['Catalog_X'], y=star['Catalog_Y'])
#             r = CirclePixelRegion(xy, radius)
#             rstr = r.serialize(format='ds9')
#             rline = rstr.strip().split('\n')[-1]
#             cmd = f'region command "{rline}"'
#             self.ds9_set(cmd)
