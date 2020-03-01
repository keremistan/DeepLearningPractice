import os
import sys
import shutil
import json
import re
from os.path import join
from PIL import Image
from torch import float32, as_tensor
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from skimage import transform
from torchvision.transforms import transforms

class Normalize(object):
    def __call__(self, sample):
        sample -= sample.mean()
        sample /= sample.std()
        return sample

class Rescale(object):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def __call__(self, sample):
        updated_sample = transform.resize(sample, self.output_shape)
        return updated_sample

class BigEarthDataSet(Dataset):
    def __init__(self, ms_images_path, transform=None):
        super(BigEarthDataSet).__init__()
        self.ms_images_path = ms_images_path
        self.transform = transform
        self.ms_images_dirs = self._get_ms_image_dirs()

    def __len__(self):
        return len(self._get_ms_image_dirs())

    def __getitem__(self, idx):
        ms_image_dir_path = join(self.ms_images_path, self.ms_images_dirs[idx])
        ms_image_dir_content = shutil.os.listdir(ms_image_dir_path)

        metadata_index = self._get_index_of_metadata_file(ms_image_dir_content)
        metadata_file = ms_image_dir_content.pop(metadata_index)
        metadata_file_path = join(ms_image_dir_path, metadata_file)
        metadata_object = []
        with open(metadata_file_path, 'r') as f:
            metadata_object = json.load(f)
        labels = metadata_object['labels']

        cur_img_as_arrays = []
        sorted_ms_image_paths = self._sort_ms_images(ms_image_dir_content)
        for img_path in sorted_ms_image_paths:
            cur_img_path = join(ms_image_dir_path, img_path)
            cur_img = Image.open(cur_img_path).getdata()
            cur_img_array = as_tensor(cur_img, dtype=float32).view(cur_img.size)
            cur_img_transformed = self.transform(cur_img_array) if self.transform else cur_img_array
            cur_img_as_arrays.append(cur_img_transformed)

        h, w = cur_img_as_arrays[0].shape
        cur_img_as_arrays = torch.as_tensor(cur_img_as_arrays).view((-1, h, w))

        return {'data': cur_img_as_arrays, 'labels': labels}

    def _get_ms_image_dirs(self):
        all_dirs = shutil.os.listdir(self.ms_images_path)
        dirs_not_starting_with_dot = [cur_dir for cur_dir in all_dirs if not cur_dir.startswith('.')]
        return dirs_not_starting_with_dot

    def _get_index_of_metadata_file(self, dir_content):
        index = 0
        for i in dir_content:
            if re.findall('.json', i).__len__() > 0:
                break
            index += 1
        return index

    def _sort_ms_images(self, ms_images):
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
        
ms_train_images_path = '/Users/base/MEGA/Universität/Tez Calismasi/dl_data/BigEarth1000/Train'
ms_test_images_path = '/Users/base/MEGA/Universität/Tez Calismasi/dl_data/BigEarth1000/Test'
composed_transforms = transforms.Compose([Rescale((120, 120)), Normalize()])

be_train_dataset = BigEarthDataSet(ms_train_images_path, composed_transforms)
be_test_dataset = BigEarthDataSet(ms_test_images_path, composed_transforms)
train_data = DataLoader(be_train_dataset)
test_data = DataLoader(be_test_dataset)

for index, sample_batch in enumerate(train_data):
    print(sample_batch)
    print('Counter: ', index)
# TODO: Train, validation ve test olarak ayir verileri.
