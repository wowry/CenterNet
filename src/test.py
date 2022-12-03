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
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt, wandb)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process),
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
  val_loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=1, 
      shuffle=False,
      drop_last=False,
      num_workers=opt.num_workers,
  )

  if 'ddu' in opt.arch:
    gmm_folderpath = f'../models/gmm/{opt.exp_id}'
    if not os.path.exists(gmm_folderpath):
      os.makedirs(gmm_folderpath)
    
    gaussians_model_path = os.path.join(gmm_folderpath, 'gaussians_model.pt')
    train_densities_path = os.path.join(gmm_folderpath, 'train_densities.pt')

    """ if opt.dataset == 'kitti':
      embeddings, labels = get_embeddings(
          detector.model,
          train_loader,
          opt.output_w,
          opt.device,
          opt.flip_test,
      )

      gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=3)
      torch.save(gaussians_model, gaussians_model_path)
    
      train_log_probs_B_Y = gmm_evaluate(detector.model, gaussians_model, train_loader, opt.output_w, opt.num_classes, opt.device)
      train_densities = torch.logsumexp(train_log_probs_B_Y, dim=-1)
      torch.save(train_densities, train_densities_path)
    else: """
    gaussians_model = torch.load(gaussians_model_path)
    train_densities = torch.load(train_densities_path)
    
    train_min_density = train_densities.min().item()
    
    gmm_dict = {
      'gaussians_model': gaussians_model,
      'train_min_density': train_min_density,
    }

  num_iters = len(dataset)
  results = {}
  logits = torch.empty(0, device=opt.device)
  densities = torch.empty(0, device=opt.device)
  epistemic_uncertainties = []
  aleatoric_uncertainties = []
  entropies = torch.empty(0, device=opt.device)
  uncs = {}
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, ((img_id, pre_processed_images), gt) in enumerate(zip(data_loader, val_loader)):
    id = img_id.numpy().astype(np.int32)[0]
    if 'ddu' in opt.arch:
      ret = detector.run(pre_processed_images, gmm_dict=gmm_dict)
      gmm_results = ret['gmm_results']
      if 'kitti' in opt.dataset \
        or ('kitti' not in opt.dataset and (False not in (gt['cls'] < 0))):
        results[id] = ret['results']
        logits = torch.cat((logits, gmm_results['logits']))

        if gmm_results['density'] is not None:
          densities = torch.cat((densities, gmm_results['density']))
        entropies = torch.cat((entropies, gmm_results['entropy']))
        epistemic_uncertainties.append(gmm_results['density'])
        aleatoric_uncertainties.append(gmm_results['entropy'])
      else:
        epistemic_uncertainties.append(None)
        aleatoric_uncertainties.append(None)
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

  #print("FPS:", 1 / avg_time_stats['tot'].avg)

  if 'ddu' in opt.arch:
    #epistemic_uncertainties = torch.stack(epistemic_uncertainties)
    #aleatoric_uncertainties = torch.stack(aleatoric_uncertainties)
    #print(epistemic_uncertainties.size(), aleatoric_uncertainties.size())
    torch.save(logits, os.path.join(gmm_folderpath, f'logits_{opt.arch}_{opt.dataset}.pt'))
    torch.save(densities, os.path.join(gmm_folderpath, f'densities_{opt.arch}_{opt.dataset}.pt'))
    torch.save(epistemic_uncertainties, os.path.join(gmm_folderpath, f'epistemic_uncertainties_{opt.arch}_{opt.dataset}.pt'))
    torch.save(aleatoric_uncertainties, os.path.join(gmm_folderpath, f'aleatoric_uncertainties_{opt.arch}_{opt.dataset}.pt'))
    torch.save(entropies, os.path.join(gmm_folderpath, f'entropies_{opt.arch}_{opt.dataset}.pt'))
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