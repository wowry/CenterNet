from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import wandb

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory, get_dataset
from detectors.detector_factory import detector_factory

from lib.utils.gmm_utils import get_embeddings, gmm_fit, gmm_evaluate

""" dataset = 'bdd'
split = 'test'
exp_id = 'bdd_centernet-2'
arch = 'dladcnddu_34'
exp_dir = f'/work/shuhei-ky/exp/CenterNet/exp/ctdet/{exp_id}'
gmm_dir = f'/work/shuhei-ky/exp/CenterNet/models/gmm/{exp_id}'
dets_path = f"{exp_dir}/results/{split}/bdd/dets" """

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      if opt.task == 'ddd':
        images[scale], meta[scale] = self.pre_process_func(
          image, scale, img_info['calib'])
      else:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  wandb.init(
    project=f'{opt.dataset}_{opt.task}',
    name=f'{opt.exp_id}_test',
    group=opt.exp_id,
    config=opt)
  Detector = detector_factory[opt.task]
  
  split = 'test' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt, wandb)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process),
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
  val_loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  prefix = ''
  should_skip = 1

  if 'ddu' in opt.arch:
    gmm_folderpath = f'../models/gmm/{opt.exp_id}'
    if not os.path.exists(gmm_folderpath):
      os.makedirs(gmm_folderpath)
    
    train_dataset = 'bdd'
    
    gaussians_model_path = os.path.join(gmm_folderpath, f'{prefix}gaussians_model_{train_dataset}.pt')

    if opt.dataset == train_dataset and not os.path.exists(gaussians_model_path):
      train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'), 
        batch_size=1, 
        shuffle=False,
        drop_last=False,
        num_workers=opt.num_workers,
      )

      embeddings, labels = get_embeddings(
          detector.model,
          train_loader,
          opt.output_w,
          opt.device,
          opt.flip_test,
      )

      gaussians_models, _ = gmm_fit(embeddings, labels, num_classes=3)
      torch.save(gaussians_models, gaussians_model_path)
    else:
      gaussians_models = torch.load(gaussians_model_path)
    
    gmm_dict = {
      'gaussians_models': gaussians_models,
    }

  num_iters = len(dataset)
  results = {}
  densities_list = []
  densities_mean_list = []
  density_bbox_list = []
  heatmaps_bbox_list = []
  heatmaps_all_list = []
  preds_overdet_list = []
  center_list = []
  size_list = []
  densities_all_0_list = []
  densities_all_1_list = []
  densities_all_2_list = []
  uncs = {}
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  has_skip = True
  for ind, ((img_id, pre_processed_images), gt) in enumerate(zip(data_loader, val_loader)):
    if should_skip == True:
      break
    has_skip = False
    id = img_id.numpy().astype(np.int32)[0]
    if 'ddu' in opt.arch:
      ret = detector.run(pre_processed_images, gmm_dict=gmm_dict, batch=gt)#, X2=heatmaps_all[ind])
      gmm_results = ret['gmm_results']
      results[id] = ret['results']
      #size_list.append(gmm_results['size'])
      #densities_list.append(gmm_results['densities_center'])
      if 'densities_mean' in gmm_results.keys():
        densities_mean_list.append(gmm_results['densities_mean'])
      #density_bbox_list.append(gmm_results['densities_bbox_list'])
      #heatmaps_bbox_list.append(gmm_results['heatmaps_bbox_list'])
      #heatmaps_all_list.append(gmm_results['heatmaps_all'])
      preds_overdet_list.append(torch.tensor(gmm_results['preds_overdet']))
      #center_list.append(gmm_results['center'])
      #densities_all_0_list.append(gmm_results['densities_all_0'])
      #densities_all_1_list.append(gmm_results['densities_all_1'])
      #densities_all_2_list.append(gmm_results['densities_all_2'])
    else:
      ret = detector.run(pre_processed_images)
      results[id] = ret['results']

    if opt.unc_est:
      uncs[id] = {k: ret['uncs'][k] for k in ret['uncs'].keys()}
      U_obj = ret['uncs']['obj']

      meta = gt['meta']
      U_obj_gt = []
      for m in meta['gt_det'][0]:
        x1, y1, x2, y2, score, cls_id = m
        ct_y, ct_x = int((y1 + y2) / 2), int((x1 + x2) / 2)
        pred = U_obj[0, :, ct_y, ct_x]
        U_obj_gt.append([int(cls_id), *pred])
      uncs[id]['obj'] = U_obj_gt
      uncs[id]['loc'] = ret['uncs']['loc'][0]
      uncs[id]['dim'] = ret['uncs']['dim'][0]
      uncs[id]['cls'] = ret['uncs']['cls'][0]
    
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()

  if not has_skip:
    print("FPS:", 1 / avg_time_stats['tot'].avg)

  if 'ddu' in opt.arch and not has_skip:
    if split == 'test':
      print("Saved preds_overdet_list")
      torch.save(torch.stack(preds_overdet_list), os.path.join(gmm_folderpath, f'{prefix}preds_overdet_list_{split}_{opt.dataset}.pt'))
    #torch.save(torch.stack(heatmaps_all_list), os.path.join(gmm_folderpath, f'{prefix}heatmaps_all_list_{split}_{opt.dataset}.pt'))
    #torch.save(heatmaps_bbox_list, os.path.join(gmm_folderpath, f'{prefix}heatmaps_bbox_list_{split}_{opt.dataset}.pt'))
    #torch.save(density_bbox_list, os.path.join(gmm_folderpath, f'{prefix}densities_bbox_list_{split}_{opt.dataset}.pt'))
    #torch.save(center_list, os.path.join(gmm_folderpath, f'{prefix}center_list_{split}_{opt.dataset}.pt'))
    #torch.save(size_list, os.path.join(gmm_folderpath, f'{prefix}size_list_{split}_{opt.dataset}.pt'))
    if len(densities_mean_list) > 0:
      torch.save(densities_mean_list, os.path.join(gmm_folderpath, f'{prefix}densities_mean_list_{split}_{opt.dataset}.pt'))
    #torch.save(densities_list, os.path.join(gmm_folderpath, f'{prefix}densities_center_{split}_{opt.dataset}.pt'))
    #torch.save(densities_all_0_list, os.path.join(gmm_folderpath, f'{prefix}densities_all_0_list_{split}_{opt.dataset}.pt'))
    #torch.save(densities_all_1_list, os.path.join(gmm_folderpath, f'{prefix}densities_all_1_list_{split}_{opt.dataset}.pt'))
    #torch.save(densities_all_2_list, os.path.join(gmm_folderpath, f'{prefix}densities_all_2_list_{split}_{opt.dataset}.pt'))
  dataset.run_eval(results, uncs, opt.save_dir, wandb)

def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])

    if opt.task == 'ddd':
      ret = detector.run(img_path, img_info['calib'])
    else:
      ret = detector.run(img_path)
    
    results[img_id] = ret['results']

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)

if __name__ == '__main__':
  opt = opts().parse()
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)