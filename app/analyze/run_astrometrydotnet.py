#!python3

## Import General Tools
from pathlib import Path
import datetime
import subprocess
from astropy import wcs

from app.data_models.OSCImage import OSCImage


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def solve_field(datamodel, cfg={}):
    assert isinstance(datamodel, OSCImage)

    cmd = [cfg['Astrometry.net'].get('solve-field','solve-field')]
    cmd.extend(['-p', '-O'])
    # index-dir
    index_dir = cfg['Astrometry.net'].get('index-dir', None)
    if index_dir: cmd.extend(['--index-dir', index_dir])
    # downsample
    downsample = cfg['Astrometry.net'].getint('downsample', None)
    if downsample: cmd.extend(['-z', f'{downsample:d}'])
    # SIP order
    SIPorder = cfg['Astrometry.net'].getint('SIPorder', None)
    if SIPorder: cmd.extend(['-t', f'{SIPorder:d}'])
    # Pixel Scale
    if 'Telescope' in cfg.sections() and 'Camera' in cfg.sections():
        fl = cfg['Telescope'].getfloat('FocalLength', None)
        pix = cfg['Camera'].getfloat('PixelSize', None)
        if fl and pix:
            pscale = 206.265*pix/fl
            cmd.extend(['-L', f'{0.95*pscale:.3f}'])
            cmd.extend(['-H', f'{1.05*pscale:.3f}'])
            cmd.extend(['-u', 'arcsecperpix'])

    # run astrometry.net on the temporary fits file
    tfile = datamodel.write_tmp()
    tfolder = tfile.parent
    cmd.append(str(tfile))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('Astrometry.net STDOUT')
    stdout_lines = proc.stdout.decode().strip().split('\n')
    for line in stdout_lines:
        print(line)
    stderr_lines = proc.stderr.decode().strip().split('\n')
    if len(stderr_lines) > 0: print('Astrometry.net STDERR')
    for line in stderr_lines:
        print(line)
    temp_contents = [f for f in tfolder.glob('*')]
    temp_files = [f.name for f in temp_contents]
    if tfile.name.replace('.fits', '.solved') in temp_files:
        wcs_file = tfolder / tfile.name.replace('.fits', '.wcs')
        new_wcs = wcs.WCS(str(wcs_file))

        # Update data model
        now = datetime.datetime.now()
        nowstr = now.strftime('%Y-%m-%D %H:%M:%S')
        datamodel.data.wcs = new_wcs
        datamodel.hdulist[1].header.set('BIASSUB', True, 'Bias subtracted')
        datamodel.hdulist[1].header.add_history('New WCS solved by astrometry.net')
        new_header = new_wcs.to_header(relax=True)
        for card in new_header.cards:
            print(card)
            datamodel.hdulist[1].header.set(card.keyword, card.value, card.comment)

    for f in temp_contents:
        print(str(f))
        f.unlink()

