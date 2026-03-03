#!python3
from pathlib import Path
import json
import re
import numpy as np

from app import log


##-------------------------------------------------------------------------
## find_files
##-------------------------------------------------------------------------
def find_files(p, pattern):
    raw_path = p / 'Raw'
    images_path = p / 'Images'
    with open(images_path / f'Stack_{pattern}.json', 'r') as jf:
        metadata = json.loads(jf.read())
    stack_count = metadata.get('Camera Info').get('Stack Count')
    exptime = float(metadata.get('Camera Info').get('Exposure (seconds)'))
    gain = metadata.get('Camera Info').get('Gain Setting')
    raw_files = [f for f in raw_path.glob(f'exp_{pattern}*.fit')]
    total_exptime = exptime*len(raw_files)
    if len(raw_files) < stack_count:
        log.warning(f'--> Found only {len(raw_files)} raw files. Stack count is {stack_count}')

    nraw = len(raw_files)
    log.info(f'  Found {exptime:.0f}s x {nraw} raw files = {total_exptime/60:.1f} min')

    framenos = []
    exptimes = []
    temperatures = []
    for raw_file in raw_files:
        patt = 'exp_'+pattern+'_([0-9]{4})_([0-9]{2})sec_([-+]?[0-9]+)C.fit'
        ismatch = re.match(patt, raw_file.name)
        if ismatch:
            framenos.append(int(ismatch.group(1)))
            exptimes.append(int(ismatch.group(2)))
            temperatures.append(int(ismatch.group(3)))

    # Temperature
    bins = np.arange(min(temperatures)-0.5, max(temperatures)+1.5, 1)
    temp_hist = np.histogram(temperatures, bins=bins)
    peak_index = np.argmax(temp_hist[0])
    temperature = int(np.mean(temp_hist[1][peak_index:peak_index+2]))
    mean_frac = max(temp_hist[0])/len(temperatures)
    std_temp = np.std(temperatures)
    log.info(f"  Typical Temperature = {temperature:d} C")
    if mean_frac < 0.99:
        log.info(f"  Fraction at Temperature = {mean_frac:.0%}")
        deltas = abs(np.array(temperatures) - temperature)
        w = deltas > 0.9
        log.warning(f"  Outlier Temps:")
        for temp_count in sorted(temp_hist[0])[:-1]:
            wtcs = np.where(temp_count == temp_hist[0])[0]
            for wtc in wtcs:
                tctemp = int(np.mean(temp_hist[1][wtc:wtc+2]))
                log.warning(f"--> {temp_count}/{nraw} files at {tctemp:d} C")

    # ExpTime
    bins = np.arange(min(exptimes)-0.5, max(exptimes)+1.5, 1)
    exptime_hist = np.histogram(exptimes, bins=bins)
    peak_index = np.argmax(exptime_hist[0])
    exptime = int(np.mean(exptime_hist[1][peak_index:peak_index+2]))
    mean_frac = max(exptime_hist[0])/len(exptimes)
    std_exptime = np.std(exptimes)
    if mean_frac < 0.99:
        log.info(f"  Fraction at Nominal ExpTime = {mean_frac:.0%}")
        deltas = abs(np.array(exptimes) - exptime)
        w = deltas > 0.9
        log.warning(f"  Outlier ExpTimes:")
        for exptime_count in sorted(exptime_hist[0])[:-1]:
            wecs = np.where(exptime_count == exptime_hist[0])[0]
            for wec in wecs:
                tcexp = int(np.mean(exptime_hist[1][wec:wec+2]))
                log.warning(f"--> {exptime_count}/{nraw} files at {tcexp:d} sec")

        log.info(np.array(exptimes)[w])

    # Dark File
    if temperature > -0.1 and temperature < 0.1:
        dark_file_name = f'StackDark_00C_{exptime:02.0f}_{gain}.fit'
    else:
        dark_file_name = f'StackDark_{temperature:.0f}C_{exptime:02.0f}_{gain}.fit'
    dark_file = p / 'DarkLibrary' / dark_file_name
    if dark_file.exists():
        log.info(f"  Dark File: {dark_file}")
    else:
        log.info(f"  Dark File: {dark_file} Does not exist!")
        dark_file = None

    result = {'RawFiles': raw_files,
              'DarkFile': dark_file,
              'FrameNos': framenos,
              'Temperature': temperature,
              }

    return result
