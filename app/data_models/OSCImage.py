from pathlib import Path
import copy
import tempfile
import datetime

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, block_reduce
import ccdproc


class OSCImage(object):
    def __init__(self, hdulist, *args, **kwargs):
        if type(hdulist) in [str, Path]:
            hdulist = Path(hdulist).expanduser().absolute()
            assert hdulist.exists()
            hdulist = fits.open(hdulist)
        self.hdulist = hdulist
        self.hdu_names = [hdu.name for hdu in self.hdulist]
        self.center_coord = None # SkyCoord of center of field (pointing)
        self.radius = None # Radius in degrees of the field (for catalog query)
        self.stars = {} # Dictionary of tables for catalog stars

        # Assume first extension is primary (Raw data)
        assert self.hdu_names[0] == 'PRIMARY'

        # Build data object
        self.data = CCDData(data=copy.deepcopy(self.hdulist[0].data),
                            meta={'APP_DM': 'OSCImage'},
                            unit='adu',
                            )
        self.split_colors()

        # Create PROCESSED extension if needed
        if 'PROCESSED' not in self.hdu_names:
            # This is a new data model
            # Build PROCESSED Extension
            header = fits.Header()
            header.set('APP_DM', 'OSCIMAGE', 'Which type of APP data model is this?')
            processed_hdu = fits.ImageHDU(data=hdulist[0].data,
                                          header=header,
                                          name='PROCESSED')
            self.hdulist.append(processed_hdu)
        else:
            processed = get_HDU('PROCESSED')
            # Check this is our data model
            assert processed.header.get('APP_DM') == 'OSCIMAGE'


    def getHDU(self, name):
        self.hdu_names = [hdu.name for hdu in self.hdulist]
        try:
            ind = self.hdu_names.index(name)
        except ValueError:
            ind = -1
        return ind


    def split_colors(self):
        self.bayer = []
        for row in range(self.data.shape[0]):
            if row%2==0: self.bayer.append(['R', 'G']*int(self.data.shape[1]/2))
            if row%2!=0: self.bayer.append(['G', 'B']*int(self.data.shape[1]/2))
        self.bayer = np.array(self.bayer)
        self.red_mask = self.bayer == 'R'
        self.green_mask = self.bayer == 'G'
        self.blue_mask = self.bayer == 'B'
        red = block_reduce(np.ma.MaskedArray(data=self.data, mask=self.red_mask),
                           2, func=np.nanmean)
        self.red = CCDData(data=red, unit='adu',
                           meta={'APP_DM': 'OSCImage', 'COLOR': 'Red'})
        green = block_reduce(np.ma.MaskedArray(data=self.data, mask=self.green_mask),
                           2, func=np.nanmean)
        self.green = CCDData(data=green, unit='adu',
                           meta={'APP_DM': 'OSCImage', 'COLOR': 'Green'})
        blue = block_reduce(np.ma.MaskedArray(data=self.data, mask=self.blue_mask),
                           2, func=np.nanmean)
        self.blue = CCDData(data=blue, unit='adu',
                           meta={'APP_DM': 'OSCImage', 'COLOR': 'Blue'})


    def update_data(self, newdata, header=[], history=[]):
        '''Update the .data attribute (the CCDData object) and add any header
        and history lines to the .hdulist[1].header
        '''
        # Image Data
        if isinstance(newdata, CCDData):
            self.data = newdata
            self.split_colors()
        # Header
        for h in header:
            print(h)
            self.hdulist[1].header.set(*h)
        # History
        now = datetime.datetime.now()
        nowstr = now.strftime('%Y-%m-%D %H:%M:%S')
        for historyline in history:
            self.hdulist[1].header.add_history(f"{nowstr}: {historyline}")


    def write_tmp(self):
        '''Write a single extension FITS file for temporary use (e.g. by
        astrometry.net)
        '''
        tdir = tempfile.mkdtemp()
        tfile = Path(tdir) / 'app_temp_image.fits'
        ccdproc.fits_ccddata_writer(self.data, tfile)
        return tfile


    def write(self, filename, overwrite=True):
        '''Write a Multi-Extension FITS file to hold the entire data model.
        '''
        assert len(self.data.to_hdu(as_image_hdu=True)) == 1
        # Processed Image Data
        pind = self.getHDU('PROCESSED')
        phdu = self.data.to_hdu(as_image_hdu=True)[0]
        phdu.header.set('EXTNAME', 'PROCESSED')
        self.hdulist[pind] = phdu
        # Header info
        cra, cdec = self.center_coord.to_string('hmsdms', sep=':', precision=2).split()
        self.hdulist[pind].header.set('CENT_RA', cra,
                                      'RA coordinate at center of FoV')
        self.hdulist[pind].header.set('CENT_DEC', cdec,
                                      'Dec coordinate at center of FoV')
        self.hdulist[pind].header.set('FOVRAD', f'{self.radius:.3f}',
                                      'Radius encompassing FoV [degrees]')

        # Three Colors
        for color in ['RED', 'GREEN', 'BLUE']:
            cind = self.getHDU(color)
            chdu = getattr(self, color.lower()).to_hdu(as_image_hdu=True)[0]
            chdu.header.set('EXTNAME', color)
            if cind == -1:
                self.hdulist.append(chdu)
            else:
                self.hdulist[cind] = chdu

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


# HDU [PRIMARY]: original raw file
# HDU [PROCESSED]: current full resolution data + standardized header
# HDU [BLUE]: blue data downsampled 2x2
# HDU [GREEN]: green data downsampled 2x2
# HDU [RED]: red data downsampled 2x2
# HDU [blue_bkg]: blue background downsampled 2x2
# HDU [green_bkg]: green background downsampled 2x2
# HDU [red_bkg]: red background downsampled 2x2
# HDU [blue_var]: blue variance downsampled 2x2
# HDU [green_var]: green variance downsampled 2x2
# HDU [red_var]: red variance downsampled 2x2
# HDU [catalog_<catalogname>]: FITS table of catalog stars
# HDU [photometry]: FITS table of photometry results
