#!python3

## Import General Tools
from pathlib import Path
import datetime
import subprocess
import numpy as np
from astropy import wcs

from app import log
from app.data_models.OSCImage import OSCImage


##-------------------------------------------------------------------------
## 
##-------------------------------------------------------------------------
def solve_field(datamodel, cfg={}, center_coord=None):
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
    # Center Coordinate
    if center_coord is not None:
        cmd.extend(['-3', f'{center_coord.ra.deg:.3f}'])
        cmd.extend(['-4', f'{center_coord.dec.deg:.3f}'])
        cmd.extend(['-5', f'0.1'])

    # run astrometry.net on the temporary fits file
    tfile = datamodel.write_tmp()
    tfolder = tfile.parent
    cmd.append(str(tfile))
    log.info('Running Astrometry.net')
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log.debug('Astrometry.net STDOUT')
    stdout_lines = proc.stdout.decode().strip().split('\n')
    for line in stdout_lines:
        log.debug(f"  {line}")
    stderr_lines = proc.stderr.decode().strip().split('\n')
    if len(stderr_lines) > 0: log.debug('Astrometry.net STDERR')
    for line in stderr_lines:
        log.debug(f"  {line}")
    temp_contents = [f for f in tfolder.glob('*')]
    temp_files = [f.name for f in temp_contents]
    if tfile.name.replace('.fits', '.solved') not in temp_files:
        center_coord = None
        radius = None
    else:
        wcs_file = tfolder / tfile.name.replace('.fits', '.wcs')
        new_wcs = wcs.WCS(str(wcs_file))
        new_header = new_wcs.to_header(relax=True)
        # Update data model
        datamodel.update_data(None,
                              header=new_header.cards,
                              history=['WCS solved by astrometry.net'])
        # Calculate and return center coordinate and field radius
        center_coord = new_wcs.pixel_to_world(datamodel.data.shape[0]/2, datamodel.data.shape[1]/2)
        center_coord_str = center_coord.to_string("hmsdms", sep=":", precision=1)
        fp = new_wcs.calc_footprint(axes=datamodel.data.shape)
        dra = fp[:,0].max() - fp[:,0].min()
        ddec = fp[:,1].max() - fp[:,1].min()
        radius = np.sqrt((dra*np.cos(fp[:,1].mean()*np.pi/180.))**2 + ddec**2)/2.
    for f in temp_contents:
        log.debug(f"Deleted temp file: {str(f)}")
        f.unlink()
    log.info(f'  Central Coordinate: {center_coord_str}')
    log.info(f'  FoV radius: {radius:.1f} deg')
    datamodel.center_coord = center_coord
    datamodel.radius = radius

