from pathlib import Path
import sys
from copy import deepcopy
import tempfile
import datetime

import numpy as np
from astropy import units as u
from astropy.io import fits

from astropy.io.fits.verify import VerifyWarning
import warnings
warnings.simplefilter('ignore', category=VerifyWarning)

from astropy.nddata import CCDData, block_reduce, block_replicate
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
    '''Object holds the data model for AstrophotoProcessing for One Shot Color
    cameras.  The data model is an object which primarily contains four CCDData
    objects: one for the full resolution image (before debayering) and one for
    each of the R, G, and B channels once debayered.

    Properties:
    hdulist - HDUList
    data - CCDData
    red - CCDData
    green - CCDData
    blue - CCDData
    reference - str: File name of reference image
    center_coord - SkyCoord

    # Image properties in self.ccd.header
    RAWFILE  - str: Raw file name
    FOVRAD   - float: Radius of FoV [deg]
    WCSMDOFF - float: Median offset between WCS and star centroid [pix]
    FWHM     - float: Median FWHM of stars [pix]
    FWHMSD   - float: Std dev of FWHM values [pix]
    ELONG    - float: Median elongation of stars
    ELONGSD  - float: Std dev of elongation values
    RA       - str # Copy of center_coord info [hh:mm:ss.s]
    DEC      - str # Copy of center_coord info [dd:mm:ss.s]
    # Color specific properties in self.ccd.header
    cZEROPT  - float: Median zero point of stars [mag]
    cZEROSD  - float: Std dev of zero point values [mag]
    cSKYB    - float: Median sky brightness around stars [ADU]
    cSKYBSD  - float: Std dev of sky brightness values [ADU]
    cBKGOFF  - float: Background offset relative to reference image [ADU]
    cSCALE   - float: Flux scaling relative to reference image [ADU]
    '''
    def __init__(self, inputval, *args, **kwargs):
        if isinstance(inputval, str):
            inputpath = Path(inputval).expanduser().absolute()
        if isinstance(inputval, Path):
            inputpath = inputval.expanduser().absolute()
        else:
            print(f"input {type(inputval)} is unknown")
            sys.exit(1)
        assert inputpath.exists()
        log.debug(f'Instantiating data model from {inputval.name}')
        hdulist = fits.open(inputval)
        hdulist.verify("fix")
        self.exptime = float(hdulist[0].header.get('EXPTIME', 0))
        self.stars = {}
        self.tempdir = Path(tempfile.mkdtemp())

        # Is is a newly opened file (single extension) or is a a processed
        # instance of an OSCImage which has been written to FITS file?
        processed = 'PROCESSED' in [hdu.name for hdu in hdulist]

        # Instantiate processed data if needed
        if not processed:
            # This is a new data model
            # Build PROCESSED Extension
            self.ccd = CCDData(data=deepcopy(hdulist[0].data), unit='adu')
            self.ccd.header['DATAMODL'] = 'OSCImage'
            self.ccd.header['COLOR'] = 'RGGB'
            self.ccd.header['RAWFILE'] = inputpath.name
            self.build_color_mask()
            self.red = None
            self.green = None
            self.blue = None
        else:
            # Read in existing data model
            pind = self.getHDU(hdulist, "PROCESSED")
            processed_data = hdulist[pind]
            mind = self.getHDU(hdulist, "MASK")
            mask_data = hdulist[mind] if mind >= 0 else None
            self.ccd = CCDData(data=processed_data,
                               mask=mask_data)
            self.build_color_mask()
            self.red = None
            self.green = None
            self.blue = None
            # Build Catalogs
            catalog_names = ['Gaia_DR3']
            for catalog in catalog_names:
                if f"CAT_{catalog}" in [hdu.name for hdu in hdulist]:
                    hduind = self.getHDU(hdulist, f"CAT_{catalog}")
                    self.stars[catalog] = Table(hdulist[hduind].data)

#         # Connect to ds9 via SAMP
#         try:
#             self.ds9 = samp.SAMPIntegratedClient()
#             self.ds9.connect()
#         except samp.errors.SAMPHubError as e:
#             self.ds9 = None
#         except Exception as e:
#             print('Unable to connect to ds9')
#             print(type(e))
#             print(e)
#             self.ds9 = None


    ##-------------------------------------------------------------------------
    ## build_color_mask
    ##-------------------------------------------------------------------------
    def build_color_mask(self):
        bayer = []
        for row in range(self.ccd.shape[0]):
            if row%2==0: bayer.append(['R', 'G']*int(self.ccd.shape[1]/2))
            if row%2!=0: bayer.append(['G', 'B']*int(self.ccd.shape[1]/2))
        bayer = np.array(bayer)
        self.mask = {'R': bayer == 'R',
                     'G': bayer == 'G',
                     'B': bayer == 'B'}


    ##-------------------------------------------------------------------------
    ## split_colors
    ##-------------------------------------------------------------------------
    def split_colors(self):
        red = block_reduce(np.ma.MaskedArray(data=self.ccd, mask=self.mask['R']),
                           2, func=np.nanmean)
        self.red = CCDData(data=block_replicate(red, 2), unit='adu',
                           wcs=self.ccd.wcs,
                           meta={'DATAMODL': 'OSCImage', 'COLOR': 'Red'})
        green = block_reduce(np.ma.MaskedArray(data=self.ccd, mask=self.mask['G']),
                           2, func=np.nanmean)
        self.green = CCDData(data=block_replicate(green, 2), unit='adu',
                           wcs=self.ccd.wcs,
                           meta={'DATAMODL': 'OSCImage', 'COLOR': 'Green'})
        blue = block_reduce(np.ma.MaskedArray(data=self.ccd, mask=self.mask['B']),
                           2, func=np.nanmean)
        self.blue = CCDData(data=block_replicate(blue, 2), unit='adu',
                           wcs=self.ccd.wcs,
                           meta={'DATAMODL': 'OSCImage', 'COLOR': 'Blue'})


    ##-------------------------------------------------------------------------
    ## getHDU
    ##-------------------------------------------------------------------------
    def getHDU(self, hdulist, name):
        hdu_names = [hdu.name for hdu in hdulist]
        try:
            ind = hdu_names.index(name)
        except ValueError:
            ind = -1
        return ind


    ##-------------------------------------------------------------------------
    ## write_tmp
    ##-------------------------------------------------------------------------
    def write_tmp(self):
        '''Write a single extension FITS file for temporary use (e.g. by
        astrometry.net)
        '''
        tfile = self.tempdir / 'app_temp_image.fits'
        ccdproc.fits_ccddata_writer(self.ccd, tfile)
        return tfile


    ##-------------------------------------------------------------------------
    ## write
    ##-------------------------------------------------------------------------
    def write(self, filename, overwrite=True):
        '''Write the data model out as one or more FITS files. The processed
        image will be one file and each of the red, green, and blue will be
        separate files.
        '''
        raw_file_name = Path(self.ccd.header.get('RAWFILE'))
        ext = raw_file_name.suffix
        filename = Path(filename).expanduser().absolute()
        processed_hdul = self.ccd.to_hdu(hdu_mask='MASK')
        processed_hdul[0].name = 'PROCESSED'
        hdunames = [hdu.header.get('EXTNAME') for hdu in processed_hdul]
        log.info(f'Output file will have {len(hdunames)} image extensions:')
        for i,hdu in enumerate(processed_hdul):
            log.info(f"  {i}: {hdunames[i]}")
        # Write Catalog Stars to FITS Table
        log.info(f'Output file will have {len(self.stars.keys())} table extensions:')
        for catalog in self.stars.keys():
            FITStable = fits.table_to_hdu(self.stars[catalog])
            FITStable.header.set('EXTNAME', f"CAT_{catalog}")
            processed_hdul.append(FITStable)
            log.info(f"  {i}: {FITStable.header.get('EXTNAME')}")
        # Write to file
        processed_hdul.writeto(filename, overwrite=overwrite)

        # Write Colors
        for color in ['red', 'green', 'blue']:
            color_ccd = getattr(self, color)
            if color_ccd is not None:
                color_hdul = color_ccd.to_hdu()
                color_hdul[0].name = color.upper()
                color_file = filename.parent / filename.name.replace(ext, f'_{color[0].upper()}{ext}')
                log.info(f'Writing {color} file: {color_file} with {len(color_hdul)} extensions')
                color_hdul.writeto(color_file, overwrite=overwrite)






    ##-------------------------------------------------------------------------
    ## write_jpg
    ##-------------------------------------------------------------------------
    def write_jpg(self, output=None, catalog='Gaia_DR3', radius=8):
        '''Overlay the catalog stars using the WCS on the grayscale image and
        generate a JPG file.
        '''
        log.info('Generating JPG image')
        stars = self.stars.get(catalog, [])

        norm = vis.ImageNormalize(self.ccd,
                                  interval=vis.AsymmetricPercentileInterval(1, 99.99),
                                  stretch=vis.LogStretch())
        plt.figure(figsize=(12,12), dpi=300)
        plt.imshow(self.ccd, cmap=plt.cm.gray, norm=norm)
        plt.xlim(0,self.ccd.shape[1])
        plt.ylim(0,self.ccd.shape[0])
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
    ## write_color_jpg
    ##-------------------------------------------------------------------------
    def write_color_jpg(self, output=None):
        '''Generate a color JPG file.
        '''
        log.info('Generating Color JPG image')
#         plt.figure(figsize=(12,12), dpi=300)
#         plt.imshow(image, origin='upper')
#         plt.xlim(0,image.shape[1])
#         plt.ylim(0,image.shape[0])
#         plt.xticks([])
#         plt.yticks([])
        # Save JPEG
        rawext = Path(self.raw_file_name).suffix
        if output is None:
            jpeg_file = Path(self.raw_file_name.replace(rawext, '_rgb.jpg'))
        elif Path(output).is_dir():
            jpeg_file = Path(output).expanduser() / self.raw_file_name.replace(rawext, '_rgb.jpg')
        else:
            jpeg_file = Path(output).expanduser()
        if jpeg_file.exists(): jpeg_file.unlink()
        log.info(f"Saving {str(jpeg_file)}")

        vis.make_rgb(self.red, self.green, self.blue,
                     interval=vis.AsymmetricPercentileInterval(1, 99.99),
                     stretch=vis.LogStretch(),
                     filename=jpeg_file)

#         plt.savefig(jpeg_file, bbox_inches='tight', pad_inches=0.1, dpi=300)




    ##-------------------------------------------------------------------------
    ## display
    ##-------------------------------------------------------------------------
#     def ds9_set(self, cmd):
#         if self.ds9:
#             msg = {"samp.mtype": "ds9.set",
#                    "samp.params":{"cmd": cmd}}
#             self.ds9.notify_all(msg)


#     def ds9_get(self, cmd):
#         if self.ds9:
#             msg = {"samp.mtype": "ds9.get",
#                    "samp.params":{"cmd": cmd}}
#             self.ds9.notify_all(msg)


#     def display_using_set(self):
#         if self.ds9:
#             tempfile = self.tempdir / 'app_temp_image.fits'
#             x, y = self.ccd.shape
#             fp = np.memmap(tempfile, dtype='float32', mode='w+', shape=self.ccd.shape)
#             fp[:] = np.array(self.ccd)[:]
#             fp.flush()
#             self.ds9_set(f"array {str(tempfile)}[xdim={x:d},ydim={y:d},bitpix=-32]")
#             tempfile.unlink()


#     def display(self):
#         if self.ds9:
#             tempfile = self.tempdir / 'app_temp_image.fits'
#             x, y = self.ccd.shape
#             fp = np.memmap(tempfile, dtype='float32', mode='w+', shape=self.ccd.shape)
#             fp[:] = np.array(self.ccd)[:]
#             fp.flush()
#             cmd = f"array {str(tempfile)}[xdim={x:d},ydim={y:d},bitpix=-32]"
#             self.ds9.ecall_and_wait("c1", "ds9.set", "10", cmd=cmd)
#             tempfile.unlink()


#     def regions_from_catalog(self, catalog='Gaia_DR3', radius=10):
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
