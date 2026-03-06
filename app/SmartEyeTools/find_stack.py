from pathlib import Path
import json
import configparser

from app import log
from app.SmartEyeTools.find_files import find_files


##-------------------------------------------------------------------------
## find_stacks
##-------------------------------------------------------------------------
def find_stack(p, objectname, min_stack_count=20):
    log.info(f'Locating files info for Object = "{objectname}"')
    object_names = configparser.ConfigParser()
    object_names_file = p / 'object_names.cfg'
    if not object_names_file.exists():
        log.warning(f'{object_names_file} does not exist!')
        object_names = {}
    else:
        object_names.read(object_names_file)

    for pattern in object_names['ObjectNames'].keys():
        if object_names['ObjectNames'].get(pattern) == objectname:
            break

    p = Path(p)
    if not p.exists():
        log.error(f'Could not find path {p}')
        return
    raw_path = p / 'Raw'
    images_path = p / 'Images'
    stacks = {}

    json_file = images_path / Path(f'Stack_{pattern}.json')

    with open(json_file, 'r') as jf:
        metadata = json.loads(jf.read())
    stack_count = metadata.get('Camera Info').get('Stack Count')
    exptime = float(metadata.get('Camera Info').get('Exposure (seconds)'))
    total_exptime = exptime*stack_count
    if stack_count < min_stack_count:
        return {}
    else:
        stack = {'ExpTime': exptime,
                 'StackCount': stack_count,
                 'TotalExpTime': total_exptime,
                 'Object': objectname}
        # Make Working Directory
        object_dir = Path(p) / objectname
        if object_dir.exists() is False:
            object_dir.mkdir(mode=0o755)
        # Get Additional Info
        stack.update(find_files(p, pattern))

        return stack
