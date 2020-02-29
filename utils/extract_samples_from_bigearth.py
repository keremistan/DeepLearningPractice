import os, random
from shutil import copytree

target_data_files_path = '/Users/base/Desktop/Data'
base_data_files_path = '/Volumes/Harddisk/BigEarth/BigEarthNet-v1.0/'
all_files_path = '/Volumes/Harddisk/BigEarth/BigEarthMetaBilgi/folder_names.txt'
snowy_files_path = '/Volumes/Harddisk/BigEarth/BigEarthMetaBilgi/patches_with_seasonal_snow.csv'
cloudy_files_path = '/Volumes/Harddisk/BigEarth/BigEarthMetaBilgi/patches_with_cloud_and_shadow.csv'

all_file_names = []
snowy_file_names = []
cloudy_file_names = []
selected_file_names = []

with open(all_files_path, 'r') as f:
    all_file_names = f.read().splitlines()

with open(snowy_files_path, 'r') as f:
    snowy_file_names = f.read().splitlines()

with open(cloudy_files_path, 'r') as f:
    cloudy_file_names = f.read().splitlines()

all_file_names = set(all_file_names) - set(snowy_file_names)
all_file_names -= set(cloudy_file_names)
all_file_names = list(all_file_names)

rand_nums = [random.randint(0, len(all_file_names)-1) for i in range(1000)]
randomly_selected_files = [all_file_names[i] for i in rand_nums]

for file in randomly_selected_files:
    src = os.path.join(base_data_files_path, file)
    dst = os.path.join(target_data_files_path, file)
    copytree(src, dst)
