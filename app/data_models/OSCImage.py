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
import ccdproc

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
                hdr_zp_std = self.hdulist[processed].header.get('ZPSTDDEV', None)
                if hdr_zp_std:
                    self.zero_point_stddev[color] = float(hdr_zp_std)

            # Read in FWHM Values
            hdr_fwhm = self.hdulist[processed].header.get('FWHM', None)
            self.fwhm = float(hdr_fwhm) if hdr_fwhm is not None else None
            hdr_fwhm_std = self.hdulist[processed].header.get('FWHMSTD', None)
            self.fwhm_stddev = float(hdr_fwhm_std) if hdr_fwhm_std is not None else None

        self.build_color_mask()

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
        tdir = tempfile.mkdtemp()
        tfile = Path(tdir) / 'app_temp_image.fits'
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
            self.hdulist[pind].header.set(f'{color}ZPSTDDEV', f'{self.zero_point_stddev.get(color):.3f}',
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
