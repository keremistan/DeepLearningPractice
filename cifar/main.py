import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pickle
from os.path import join, split
from os import getcwd
from pprint import pprint
import numpy as np
from matplotlib import pyplot as pl

# TODO: Batch Normalization
# TODO: Babysitting the learning process
# TODO: Hyperparameter Optimization

# Preprocessing the data
target_parent_dir = '/Users/base/MEGA/Universität/Tez Calismasi/'
target_base_dir = 'dl_data/cifar-10-batches-py/'

def read_data(file_path):
    with open(file_path, 'rb') as data_reading:
        data = pickle.load(data_reading, encoding='latin1')
        data_reading.close()
        return data

def display_image(rgb):
    print(rgb)
    pl.figure()
    pl.imshow(rgb)
    pl.show()

batch_1_path = join(target_parent_dir ,target_base_dir, 'data_batch_1')
test_path = join(target_parent_dir, target_base_dir, 'test_batch')
batch1 = read_data(batch_1_path)
test_batch = read_data(test_path)

kullanilacak_miktar = 100
training_data = batch1['data'][:kullanilacak_miktar]
training_label = batch1['labels'][:kullanilacak_miktar]
test_data = test_batch['data'][:2*kullanilacak_miktar]
test_label = test_batch['labels'][:2*kullanilacak_miktar]

training_data, training_label, test_data, test_label = map(torch.tensor, (training_data, training_label, test_data, test_label))

train_ds = TensorDataset(training_data, training_label)
train_dl = DataLoader(train_ds)

test_ds = TensorDataset(test_data, test_label)
test_dl = DataLoader(train_ds)

class CifarClassifier(nn.Module):
    def __init__(self):
        super(CifarClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 11, 5, padding=2)
        self.conv2 = nn.Conv2d(11, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 11, 3, padding=1)
        self.linear1 = nn.Linear(11*32*32, 10)
    
    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view((1, -1))
        x = self.linear1(x)
        return x

classifier = CifarClassifier()
mse_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

epochs = 3
for epoch in range(epochs):
    for x_img, y_label in train_dl:
        prediction = classifier(x_img)
        loss = mse_loss(prediction, y_label)
        print('Current Loss: ', loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        for x_test, y_test in test_dl:
            test_tahmini = classifier(x_test)
            kayip = mse_loss(test_tahmini, y_test)
            print('Guncel Kayip: ', kayip)