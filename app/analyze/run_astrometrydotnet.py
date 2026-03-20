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
def solve_field(DM, cfg={}, center_coord=None, search_radius=0.25):
    assert isinstance(DM, OSCImage)

    cmd = [cfg['Astrometry.net'].get('solve-field','solve-field')]
    cmd.extend(['-p', '-O'])
    cmd.extend(['--cpulimit', '100'])
    # index-dir
    index_dir = cfg['Astrometry.net'].get('index-dir', None)
    if index_dir:
        index_dir = Path(index_dir).expanduser().absolute()
        cmd.extend(['--index-dir', str(index_dir)])
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
        cmd.extend(['-5', f'{search_radius:.2f}'])

    # run astrometry.net on the temporary fits file
    tfile = DM.write_tmp()
    tfolder = tfile.parent
    cmd.append(str(tfile))
    log.info('Running Astrometry.net')
    log.debug(' '.join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log.debug('Astrometry.net STDOUT')
    stdout_lines = proc.stdout.decode().strip('\n').strip().split('\n')
    for line in stdout_lines:
        log.debug(f"  {line}")
    log.info(f"Astrometry.net STDOUT: {stdout_lines[-1]}")
    stderr_lines = proc.stderr.decode().strip('\n').strip().split('\n')
    if len(''.join(stderr_lines)) > 0:
        log.error('Astrometry.net STDERR')
        for line in stderr_lines:
            log.error(f"  {line}")
    temp_contents = [f for f in tfolder.glob('*')]
    temp_files = [f.name for f in temp_contents]
    if tfile.name.replace('.fits', '.solved') not in temp_files:
        log.warning('No solution from astrometry.net')
        center_coord = None
        radius = None
    else:
        log.debug('Found solved file from astrometry.net')
        wcs_file = tfolder / tfile.name.replace('.fits', '.wcs')
        # Update data model
        DM.ccd.wcs = wcs.WCS(str(wcs_file))
        # Calculate and return center coordinate and field radius
        center_coord = DM.ccd.wcs.pixel_to_world(DM.ccd.shape[0]/2, DM.ccd.shape[1]/2)
        fp = DM.ccd.wcs.calc_footprint(axes=DM.ccd.shape)
        dra = fp[:,0].max() - fp[:,0].min()
        ddec = fp[:,1].max() - fp[:,1].min()
        radius = np.sqrt((dra*np.cos(fp[:,1].mean()*np.pi/180.))**2 + ddec**2)/2.
    for f in temp_contents:
        log.debug(f"Deleted temp file: {str(f)}")
        f.unlink()
    if center_coord is not None:
        center_coord_str = center_coord.to_string("hmsdms", sep=":", precision=1)
        log.info(f'  Central Coordinate: {center_coord_str}')
        log.info(f'  FoV radius: {radius:.1f} deg')
        DM.center_coord = center_coord
        DM.ccd.meta['FOVRAD'] = radius
        DM.ccd.meta['RA'] = center_coord_str.split()[0]
        DM.ccd.meta['DEC'] = center_coord_str.split()[1]
    return center_coord

