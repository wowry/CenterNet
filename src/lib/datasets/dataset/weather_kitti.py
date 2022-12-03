from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math

import torch.utils.data as data
import tools.kitti_eval.tool.kitti_common as kitti
from tools.kitti_eval.tool.eval_auroc import get_official_eval_result
from tools.unc_eval.utils import get_unc_files

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

class WeatherKitti(data.Dataset):
  num_classes = 3
  default_resolution = [384, 1280]
  mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
  std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(WeatherKitti, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'kitti')
    self.img_dir = os.path.join(self.data_dir, '../weather_kitti/data_object/training/image_2/rain', '100mm', 'rainy_image')
    if opt.trainval:
      split = 'trainval' if split == 'train' else 'test'
      self.img_dir = os.path.join(self.data_dir, 'images', split)
      self.annot_path = os.path.join(
        self.data_dir, 'annotations', 'kitti_{}.json').format(split)
    else:
      self.annot_path = os.path.join(self.data_dir, 
        'annotations', 'kitti_{}_{}.json').format(opt.kitti_split, split)
    self.max_objs = 50
    self.class_name = [
      '__background__', 'Pedestrian', 'Car', 'Cyclist']
    self.cat_ids = {1:0, 2:1, 3:2, 4:-1, 5:-3, 6:-10, 7:-3, 8:-3, 9:-99}
    
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt
    self.alpha_in_degree = False

    print('==> initializing kitti {}, {} data.'.format(opt.kitti_split, split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def __len__(self):
    return self.num_samples

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    pass

  def save_results(self, results):
    if not os.path.exists(self.results_dir):
      os.makedirs(self.results_dir)
    for img_id in results.keys():
      out_path = os.path.join(self.results_dir, '{:06d}.txt'.format(img_id))
      f = open(out_path, 'w')
      for cls_ind in results[img_id]:
        for j in range(len(results[img_id][cls_ind])):
          class_name = self.class_name[cls_ind]
          f.write('{} 0.0 0 -10'.format(class_name))
          for i in range(len(results[img_id][cls_ind][j])-1):
            f.write(' {:.2f}'.format(results[img_id][cls_ind][j][i]))
          f.write(' 0.0 0.0 0.0 0.0 0.0 0.0 0.0 {:.2f}'.format(results[img_id][cls_ind][j][i+1]))
          f.write('\n')
      f.close()
  
  def save_uncs(self, image_uncs):
    for img_id in image_uncs.keys():
      uncs = image_uncs[img_id]

      for key in uncs.keys():
        unc = uncs[key]
        unc_dir = os.path.join(self.uncs_dir, key)

        if not os.path.exists(unc_dir):
          os.makedirs(unc_dir)
      
        out_path = os.path.join(unc_dir, '{:06d}.txt'.format(img_id))
        f = open(out_path, 'w')
        for i in range(len(unc)):
          for u in unc[i]:
            f.write('{:.5f} '.format(u))
          f.write('\n')
        f.close()

  def run_eval(self, results, uncs, save_dir, wandb):
    self.results_dir = os.path.join(save_dir, f'results/{self.opt.dataset}/dets')
    self.save_results(results)
    
    self.uncs_dir = os.path.join(save_dir, f'results/{self.opt.dataset}/uncs')
    self.save_uncs(uncs)
    
    det_path = self.results_dir
    dt_annos = kitti.get_label_annos(det_path)

    gt_path = os.path.join(self.opt.data_dir, 'kitti/training/label_2')
    gt_split_file = os.path.join(gt_path, '../../ImageSets_3dop/val.txt')
    val_image_ids = _read_imageset_file(gt_split_file)

    id_label_info = {
      'dataset': 'kitti',
      'id_classes': (0, 1, 2, 3, 5)
    }
    gt_annos = kitti.get_label_annos(gt_path, val_image_ids)

    if self.opt.unc_est:
      uncs = get_unc_files(self.uncs_dir)
    
    result = get_official_eval_result(gt_annos, dt_annos, uncs, (0, 1, 2), self.opt.dataset, wandb)

    ap_file = os.path.join(save_dir, f'results_ap_{self.opt.dataset}.txt')
    with open(ap_file, "w") as f:
      f.write(result)
    print(result)
