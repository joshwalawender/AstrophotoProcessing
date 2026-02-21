import configparser

from app.data_models.OSCImage import OSCImage
from app.reduce.bias_subtract import bias_subtract
from app.analyze.run_astrometrydotnet import solve_field


##-------------------------------------------------------------------------
## Define Configuration
##-------------------------------------------------------------------------
cfg = configparser.ConfigParser()
cfg.read_string('''
[Settings]
cleanup = False

[Telescope]
FocalLength = 814
Aperture = 102

[Camera]
PixelSize = 3.76

[Astrometry.net]
solve-field = /opt/homebrew/bin/solve-field
index-dir = /Volumes/Ohina2External/astrometry_data
downsample = 2
SIPorder = 4
''')


##-------------------------------------------------------------------------
## Example Reduction
##-------------------------------------------------------------------------
input_file = '/Volumes/SmartEyeData/SmartEye_2026-02-05/Raw/exp_000187_0005_0011_30sec_0C.fit'
input_image = OSCImage(input_file)

print('Blue')
print(type(input_image.blue), input_image.blue.shape)
# print(input_image.blue_mask)
print(input_image.blue)
print()
print('Green')
print(type(input_image.green), input_image.green.shape)
# print(input_image.green_mask)
print(input_image.green)
print()
print('Red')
print(type(input_image.red), input_image.red.shape)
# print(input_image.red_mask)
print(input_image.red)


# master_bias_file = '/Volumes/SmartEyeData/SmartEye_2026-02-05/DarkLibrary/StackDark_00C_30_350.fit'
# master_bias = OSCImage(master_bias_file)
# bias_subtract(input_image, master_bias=master_bias)
# for card in input_image.hdulist[1].header.cards:
#     print(card)
# 
# solve_field(input_image, cfg=cfg)