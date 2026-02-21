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
            hdulist = Path(hdulist)
            assert hdulist.exists()
            hdulist = fits.open(hdulist)
        self.hdulist = hdulist
        self.hdu_names = [hdu.name for hdu in self.hdulist]

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


    def get_HDU(self, name):
        self.hdu_names = [hdu.name for hdu in self.hdulist]
        return self.hdu_names.index(name)


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
