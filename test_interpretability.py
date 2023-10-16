# %%
import pickle
import numpy as np
import os

from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import cv2
from train_csr import ImageDataset, OpensetRecognizer
import random

data_path = 'data/'
openset_model = OpensetRecognizer(data_path='data/', feature_dim=128)

print('Train the model')
losses, accs = openset_model.train_model(n_epochs=50)
json.dump({'accs': accs, 'losses': losses}, open('stats.json', 'w'))

# load a model if already trained
# openset_model.load_model('models/35') 

test_data = pickle.load(open(os.path.join(data_path, 'test_data.pickle'), 'rb'))
test_labels = pickle.load(open(os.path.join(data_path, 'test_labels.pickle'), 'rb'))

open_data = pickle.load(open(os.path.join(data_path, 'opendata.pickle'), 'rb'))
open_labels = pickle.load(open(os.path.join(data_path, 'open_labels.pickle'), 'rb'))

open_labels_set = sorted(list(set(open_labels)))
test_labels_set = sorted(list(set(test_labels)))
test_labels_mapping = {test_labels_set[i]:i for i in range(len(test_labels_set))}
open_labels_mapping = {open_labels_set[i]:i + len(test_labels_set) for i in range(len(open_labels_set))}

test_data = np.stack(test_data)
open_data = np.stack(open_data)
test_labels = np.array([test_labels_mapping[item] for item in test_labels])
open_labels = np.array([open_labels_mapping[item] for item in open_labels])

test_dataset = ImageDataset(test_data, test_labels, n_classes=7, ls_eps=0, transform=openset_model.val_transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, drop_last=False, collate_fn=openset_model.val_collate)

open_dataset = ImageDataset(open_data, test_labels, n_classes=7, ls_eps=0, transform=openset_model.val_transform)
open_loader = DataLoader(open_dataset, batch_size=4, shuffle=False, drop_last=False, collate_fn=openset_model.val_collate)

openset_model.prepare_detection_model()

n = len(test_dataset)
m = len(open_dataset)
i, j = random.sample(range(n), 1)[0], random.sample(range(m), 1)[0]
print(i, j)

tensor_test, _ = test_dataset[i]
tensor_open, _ = open_dataset[j]
img_test, label_test = test_dataset.view(i)
img_open, label_open = open_dataset.view(j)

_ = openset_model.interpret(img_test, tensor_test, 'fig_test.png', 'GradCAM')

_ = openset_model.interpret(img_open, tensor_open, 'fig_open.png', 'GradCAM')
