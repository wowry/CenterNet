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

dataset = 'bdd'
exp_id = 'bdd_centernet'
arch = 'dladcnddu_34'
exp_dir = f'/work/shuhei-ky/exp/CenterNet/exp/ctdet/{exp_id}'
results_dir = f'{exp_dir}/results/bdd/dets'
gmm_dir = f'/work/shuhei-ky/exp/CenterNet/models/gmm/{exp_id}'

is_overdet = torch.load(f"{exp_dir}/is_overdet_{dataset}.pt")
scores_list = torch.load(f"{exp_dir}/scores_list_{dataset}.pt")
densities_bbox_list = torch.load(f"{gmm_dir}/densities_bbox_list_{arch}_{dataset}.pt")

threshold = 0.15
HEIGHT = 128
WIDTH = 224

X_pos = [[], [], []]
y_pos = []

for (densities_layers) in densities_bbox_list: # 10000
  for i, (densities, is_nodet, scores) in enumerate(zip(densities_layers, is_overdet, scores_list)): # 3
    for (density, nodet, score) in zip (densities, is_nodet, scores): # 100
      if float(score) >= threshold and density is not None and nodet.item() == 1:
        h, w = density.shape
        hp = int((WIDTH - w) / 2)
        vp = int((HEIGHT - h) / 2)
        padding = (hp, WIDTH - (w + hp), vp, HEIGHT - (h + vp))
        X_pos[i].append(F.pad(density, padding, 'constant', 0))
        
        if i == 0:
            y_pos.append(nodet.unsqueeze(-1))
