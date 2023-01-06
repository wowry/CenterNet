from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import A

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data
import tools.kitti_eval.tool.kitti_common as kitti
from tools.kitti_eval.tool.eval2 import get_official_eval_result
from tools.unc_eval.utils import get_unc_files, eval_encs

val_size = 8000

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

class BDD(data.Dataset):
  num_classes = 3
  default_resolution = [512, 896]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(BDD, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'bdd100k')
    self.img_dir = os.path.join(self.data_dir, f'images/100k/{split}' if split != 'test' else 'images/100k/val')

    if split == 'test':
      self.annot_path = os.path.join(
          self.data_dir, 'labels', 
          'val.json')
    else:
      if opt.task == 'exdet':
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_extreme_{}2017.json').format(split)
      else:
        self.annot_path = os.path.join(
          self.data_dir, 'labels', 
          '{}.json').format(split)
    self.max_objs = 128
    self.class_name = [
      '__background__', 'pedestrian', 'car', 'rider']
    self.cat_ids = {
      1: 0, 2: 2, 3: 1, 4: -3, 5: -3, 6: -3, 7: -3, 8: -3, 9: -3, 10: -3
    }
    """ self.class_name = [
      '__background__', 'pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
      'motorcycle', 'bicycle', 'trafficlight', 'trafficsign']
    self.cat_ids = {
      1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9
    } """
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    print('==> initializing BDD {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)

    if split == 'val':
      self.images = self.coco.getImgIds()[:val_size]
    elif split == 'test':
      self.images = self.coco.getImgIds()[val_size:]
    else:
      self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = (cls_ind - 1) - 1
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results_dir, results):
    if not os.path.exists(results_dir):
      os.makedirs(results_dir)
    for img_id in results.keys():
      out_path = os.path.join(results_dir, '{:06d}.txt'.format(img_id))
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
  
  def save_uncs(self, uncs_dir, image_uncs):
    for img_id in image_uncs.keys():
      uncs = image_uncs[img_id]

      for key in uncs.keys():
        unc = uncs[key]
        unc_dir = os.path.join(uncs_dir, key)

        if not os.path.exists(unc_dir):
          os.makedirs(unc_dir)
      
        out_path = os.path.join(unc_dir, '{:06d}.txt'.format(img_id))
        f = open(out_path, 'w')
        for i in range(len(unc)):
          for u in unc[i]:
            f.write('{:.5f} '.format(u))
          f.write('\n')
        f.close()
    
  def convert_labels_to_kitti_format(self, gt_path, gt_json_path):
    if not os.path.exists(gt_path):
      print("Converting labels to kitti format...")
      os.mkdir(gt_path)

      with open(gt_json_path, 'r') as f:
        gt_file = json.load(f)
      gt_cat_names = [c['name'] for c in gt_file['categories']]
      gt_raw_annos = gt_file['annotations']

      for gt_ann in gt_raw_annos:
        img_id = gt_ann['image_id']
        out_path = os.path.join(gt_path, '{:06d}.txt'.format(img_id))
        with open(out_path, "a") as f:
          class_name = gt_cat_names[gt_ann['category_id'] - 1].replace(' ', '')
          f.write('{} 0.0 0 -10'.format(class_name))
          bbox = gt_ann['bbox']
          bbox[2] += bbox[0]
          bbox[3] += bbox[1]
          for i in range(len(bbox)):
            f.write(' {:.2f}'.format(bbox[i]))
          f.write(' 0.0 0.0 0.0 0.0 0.0 0.0 0.0')
          f.write('\n')
  
  def run_eval(self, results, uncs, save_dir, wandb):
    results_dir = os.path.join(save_dir, f'results/{self.split}/{self.opt.dataset}/dets')
    self.save_results(results_dir, results)
    
    uncs_dir = os.path.join(save_dir, f'results/{self.opt.dataset}/uncs')
    self.save_uncs(uncs_dir, uncs)

    det_path = results_dir
    dt_annos = kitti.get_label_annos(det_path)

    gt_path = os.path.join(self.opt.data_dir, 'bdd100k/labels/labels')
    gt_json_path = os.path.join(gt_path, '../val.json')

    if self.split == 'test':
      val_image_ids = list(range(val_size + 1, val_size + self.num_samples + 1))
    else:
      val_image_ids = list(range(1, self.num_samples + 1))

    self.convert_labels_to_kitti_format(gt_path, gt_json_path)
    
    id_label_info = {
      'dataset': 'bdd',
      'id_classes': (0, 1, 2)
    }
    gt_annos = kitti.get_label_annos(gt_path, val_image_ids)

    if self.opt.unc_est:
      uncs = get_unc_files(uncs_dir)
    
    result = get_official_eval_result(gt_annos, dt_annos, uncs, (0, 1, 2), self.opt, wandb, difficulties=[2])

    """ ap_file = os.path.join(save_dir, f'results_ap_{self.opt.dataset}.txt')
    with open(ap_file, "w") as f:
      f.write(result)
    print(result) """
