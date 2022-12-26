import torch
from torch import nn
import torch.nn.functional as F
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from matplotlib import rc
from torchvision.models import mobilenetv2

dataset = 'kitti'
exp_id = 'kitti_centernet-6'
arch = 'dladcnddu_34'
exp_dir = f'/work/shuhei-ky/exp/CenterNet/exp/ctdet/{exp_id}'
gmm_dir = f'/work/shuhei-ky/exp/CenterNet/models/gmm/{exp_id}'

classnames = ['Pedestrian', "Car", "Cyclist"]

is_overdet_test = torch.load(f"{exp_dir}/is_overdet_test_{dataset}.pt")
scores_test_list = torch.load(f"{exp_dir}/scores_list_test_{dataset}.pt")
classes_test_list =  torch.load(f"{exp_dir}/classes_list_test_{dataset}.pt")

densities_test_bbox_list = torch.load(f"{gmm_dir}/densities_bbox_list_test_{dataset}.pt")
heatmaps_test_bbox_list = torch.load(f"{gmm_dir}/heatmaps_bbox_list_test_{dataset}.pt")

threshold = 0.15

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, data, label):
    self.data = data
    self.label = label
  
  def __len__(self):
    return len(self.label)

  def __getitem__(self, i):
    return self.data[i].float(), self.label[i]


net = torch.load(f'{gmm_dir}/overdet_pred_model.pth')

X = [[], [], []]
y = []
heatmaps = []

HEIGHT = 96
WIDTH = 96

for (densities_layers, heatmaps_list, is_nodet, scores, classes) in zip(densities_test_bbox_list, heatmaps_test_bbox_list, is_overdet_test, scores_test_list, classes_test_list):
  for i, densities in enumerate(densities_layers):
    for (density, heatmap, nodet, score, cls) in zip (densities, heatmaps_list, is_nodet, scores, classes):
      if float(score) >= threshold and density is not None:
        h, w = density.shape
        hp = int((WIDTH - w) / 2)
        vp = int((HEIGHT - h) / 2)
        padding_density = (hp, WIDTH - (w + hp), vp, HEIGHT - (h + vp))
        
        cls_int = classnames.index(cls)
        hm = heatmap[cls_int]
        h, w = hm.shape
        hp = int((WIDTH - w) / 2)
        vp = int((HEIGHT - h) / 2)
        padding_hm = (hp, WIDTH - (w + hp), vp, HEIGHT - (h + vp))
        
        X[i].append(F.pad(density, padding_density, 'constant', 0))
        
        if i == 0:
          heatmaps.append(F.pad(hm, padding_hm, 'constant', 0))
          y.append(nodet.unsqueeze(-1))

""" X = torch.stack([torch.stack(x) for x in X])
y = torch.stack(y)
heatmaps = torch.stack(heatmaps)

for i in range(3):
    X[i] = (X[i] - X[i].min())/(X[i].max() - X[i].min())

X = X.permute(1, 0, 2, 3)

test_dataset = MyDataset(X, y)
testloader = torch.utils.data.DataLoader(test_dataset)

outputs_list = []
labels_list = []

for (inputs, labels) in testloader:
  outputs = net(inputs)
  
  outputs_list.append(torch.sigmoid(outputs).reshape(-1))
  labels_list.append(labels.reshape(-1))

torch.save(outputs_list, f'{gmm_dir}/outputs_list.pth')
print(len(outputs_list)) """