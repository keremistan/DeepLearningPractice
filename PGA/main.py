import os
import sys
import shutil
import json
import re
from os.path import join
from PIL import Image
import torch
import numpy as np

def get_index_of_metadata_file(dir_content):
    index = 0
    for i in dir_content:
        if re.findall('.json', i).__len__() > 0:
            break
        index += 1
    return index

def sort_ms_images(ms_images):
    ms_images_with_band = [[re.findall('B(\d\w|\d+)', img)[0], img] for img in ms_images]
    ms_images_with_band_as_int = []
    for img_band in ms_images_with_band:
        if img_band[0] == '8A':
            img_band[0] = 8.5
        else:
            img_band[0] = int(img_band[0])

    ms_images_with_band.sort(key=lambda elem: elem[0])
    sorted_images = [img[1] for img in ms_images_with_band]
    return sorted_images


ms_images_path = '/Users/base/MEGA/Universität/Tez Calismasi/dl_data/BigEarth1000'

ms_image_labels = []
ms_images = []

ms_images_dirs = shutil.os.listdir(ms_images_path)
for cur_dir in ms_images_dirs:
    if cur_dir.startswith('.'):
        break
    cur_dir_path = join(ms_images_path, cur_dir)
    cur_dir_content = os.listdir(cur_dir_path)
    print(cur_dir_content)

    metadata_index = get_index_of_metadata_file(cur_dir_content)
    metadata_file = cur_dir_content.pop(metadata_index)
    metadata_file_path = join(cur_dir_path, metadata_file)
    metadata_object = []
    with open(metadata_file_path, 'r') as f:
        metadata_object = json.load(f)
    ms_image_labels.append(metadata_object['labels'])

    sorted_ms_image_paths = sort_ms_images(cur_dir_content)
    for img_path in sorted_ms_image_paths:
        cur_img_path = join(cur_dir_path, img_path)
        cur_img = Image.open(cur_img_path).getdata()
        cur_img_tensor = torch.tensor(cur_img)
        print(cur_img_tensor)