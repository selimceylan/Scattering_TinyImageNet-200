import glob
import os
from shutil import move
from os import listdir, rmdir

target_folder = '../Scattering/data-tiny/tiny-imagenet-200/val/'
test_folder = '../Scattering/data-tiny/tiny-imagenet-200/val_prepared/'

os.mkdir(test_folder)
val_dict = {}
with open('../Scattering/data-tiny/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]

paths = glob.glob('../Scattering/data-tiny/tiny-imagenet-200/val/images/*')
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(test_folder + str(folder)):
        os.mkdir(test_folder + str(folder))

for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]

    dest = test_folder + str(folder) + "/" +str(file)
    move(path, dest)

rmdir('../Scattering/data-tiny/tiny-imagenet-200/val/images')