from pathlib import Path
import sys
import copy
import tempfile
import datetime

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData, block_reduce
from astropy.table import Table
from astropy.coordinates import SkyCoord, ICRS
from astropy import wcs
from astropy import visualization as vis
from astropy import samp
import ccdproc
from regions import Regions, PixCoord, CirclePixelRegion

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
            log.info(f'Instantiating data model from {self.raw_file_name}')
            hdulist = fits.open(hdulist)
        else:
            print(f"HDUList input {type(hdulist)} is unknown")
            sys.exit(1)
        self.hdulist = hdulist
        self.hdulist.verify("fix")
        self.hdu_names = [hdu.name for hdu in self.hdulist]
        self.zero_point = {}
        self.zero_point_stddev = {}
        self.fwhm = None
        self.fwhm_stddev = None
        self.tempdir = Path(tempfile.mkdtemp())

        # Connect to ds9 via SAMP
        try:
            self.ds9 = samp.SAMPIntegratedClient()
            self.ds9.connect()
        except Exception as e:
            print('Unable to connect to ds9')
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

            # Read in Zero Point Values
            for color in ['R', 'G', 'B']:
                hdr_zp = self.hdulist[processed].header.get(f'{color}ZEROPNT', None)
                if hdr_zp:
                    self.zero_point[color] = float(hdr_zp)
                hdr_zp_std = self.hdulist[processed].header.get(f'{color}ZPSTD', None)
                if hdr_zp_std:
                    self.zero_point_stddev[color] = float(hdr_zp_std)

            # Read in FWHM Values
            hdr_fwhm = self.hdulist[processed].header.get('FWHM', None)
            self.fwhm = float(hdr_fwhm) if hdr_fwhm is not None else None
            hdr_fwhm_std = self.hdulist[processed].header.get('FWHMSTD', None)
            self.fwhm_stddev = float(hdr_fwhm_std) if hdr_fwhm_std is not None else None

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
#             self.split_colors()
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
        for color in self.zero_point.keys():
            self.hdulist[pind].header.set(f'{color}ZEROPNT', f'{self.zero_point.get(color):.3f}',
                                          f'Calculated Zero Point ({color})')
            self.hdulist[pind].header.set(f'{color}ZPSTD', f'{self.zero_point_stddev.get(color):.3f}',
                                          f'Calculated Zero Point Std Dev ({color})')
        if self.fwhm:
            self.hdulist[pind].header.set('FWHM', f'{self.fwhm:.2f}',
                                          'Typical FWHM')
            self.hdulist[pind].header.set('FWHMSTD', f'{self.fwhm_stddev:.2f}',
                                          'FWHM Std Dev')

        # Three Colors
#         for color in ['RED', 'GREEN', 'BLUE']:
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
    def write_jpg(self, radius=8):
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
        ext = Path(self.raw_file_name).suffix
        jpeg_file = Path(self.raw_file_name.replace(ext, '.jpg'))
        if jpeg_file.exists(): jpeg_file.unlink()
        log.info(f"Saving {str(jpeg_file)}")
        plt.savefig(jpeg_file, bbox_inches='tight', pad_inches=0.1, dpi=300)


    ##-------------------------------------------------------------------------
    ## display
    ##-------------------------------------------------------------------------
    def ds9_set(self, cmd):
        msg = {"samp.mtype": "ds9.set",
               "samp.params":{"cmd": cmd}}
        self.ds9.notify_all(msg)


    def ds9_get(self, cmd):
        msg = {"samp.mtype": "ds9.get",
               "samp.params":{"cmd": cmd}}
        self.ds9.notify_all(msg)


#     def display_using_set(self):
#         if self.ds9:
#             tempfile = self.write_tmp()
#             x, y = self.data.shape
#             self.ds9_set(f"array {str(tempfile)}[xdim={x:d},ydim={y:d},bitpix=-32]")
#             tempfile.unlink()


    def display(self):
        if self.ds9:
            tempfile = self.write_tmp()
            x, y = self.data.shape
            cmd = f"array {str(tempfile)}[xdim={x:d},ydim={y:d},bitpix=-32]"
            fp = np.memmap(tempfile, dtype='float32', mode='w+', shape=self.data.shape)
            fp[:] = np.array(self.data)[:]
            fp.flush()
            self.ds9.ecall_and_wait("c1", "ds9.set", "10", cmd=cmd)
            tempfile.unlink()


    def regions_from_catalog(self, catalog='Gaia DR3', radius=10):
        xys = [PixCoord(x=star['Catalog_X'], y=star['Catalog_Y']) for star in self.stars.get(catalog, [])]
        reglist = Regions([CirclePixelRegion(xy, radius) for xy in xys])
        reglist.write('tmp.reg')
        cmd = f'region load /Users/jwalawender/git/AstrophotoProcessing/tmp.reg'
        self.ds9_set(cmd)
#         for star in self.stars.get(catalog, []):
#             xy = PixCoord(x=star['Catalog_X'], y=star['Catalog_Y'])
#             r = CirclePixelRegion(xy, radius)
#             rstr = r.serialize(format='ds9')
#             rline = rstr.strip().split('\n')[-1]
#             cmd = f'region command "{rline}"'
#             self.ds9_set(cmd)
