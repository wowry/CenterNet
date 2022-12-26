from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
from torchvision.transforms.functional import crop
import torch.nn.functional as F
from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger
from lib.utils.gmm_utils import get_embeddings, gmm_fit, gmm_forward
import utils.uncertainty_confidence as uncertainty_confidence

HEIGHT = 96
WIDTH = 96

net = torch.load(f'/work/shuhei-ky/exp/CenterNet/models/gmm/kitti_centernet-6/overdet_pred_model.pth')

X_min = []
X_max = []

for i in range(3):
  X_min.append(torch.load(f"/work/shuhei-ky/exp/CenterNet/models/gmm/kitti_centernet-6/X{i}_min.pt"))
  X_max.append(torch.load(f"/work/shuhei-ky/exp/CenterNet/models/gmm/kitti_centernet-6/X{i}_max.pt"))

class BaseDetector(object):
  def __init__(self, opt, wandb):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(opt.arch, opt.heads, opt.head_conv, opt)
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.to(opt.device)
    self.model.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True

  def pre_process(self, image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results):
   raise NotImplementedError

  def run(self, image_or_path_or_tensor, gmm_dict=None, batch=None, meta=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
    start_time = time.time()
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)

    opt = self.opt
    outputs = []
    detections = []
    indices = []
    hms = []
    whs = []
    uncertainties = []
    for scale in self.scales:
      scale_start_time = time.time()
      if not pre_processed:
        images, meta = self.pre_process(image, scale, meta)
      else:
        # import pdb; pdb.set_trace()
        images = pre_processed_images['images'][scale][0] # 2, 3, 384, 1280
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
      images = images.to(opt.device)
      torch.cuda.synchronize()
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time
      
      output, dets, inds, hm, wh, uncs, forward_time = self.process(images, return_time=True)
      outputs.append(output)
      hms.append(hm)

      torch.cuda.synchronize()
      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      if opt.debug >= 2:
        self.debug(debugger, images, dets, output, scale)
      
      dets, uncs = self.post_process(dets, uncs, meta, scale)
      torch.cuda.synchronize()
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(dets)
      indices.append(inds)
      whs.append(wh)
      uncertainties.append(uncs)
    
    results = self.merge_outputs(detections)
    output = outputs[0] # TODO: move into merge_outputs
    indices = indices[0] # TODO: move into merge_outputs
    hms = hms[0] # TODO: move into merge_outputs
    whs = whs[0] # TODO: move into merge_outputs
    uncertainties = uncertainties[0] # TODO: move into merge_outputs

    torch.cuda.synchronize()
    
    gmm_results = {}
    if gmm_dict is not None:
      device = opt.device
      output_w = opt.output_w
      gaussians_models = gmm_dict['gaussians_models']
      #train_min_density = gmm_dict['train_min_density']
      hm = batch['all_classes_hm'].to(device)
      ind = batch['ind'][0].to(device)
      cls = batch['cls'][0].to(device)

      hm = torch.sum(hm[0, :, :, :], dim=0)
      inds_bg_r, inds_bg_c = torch.where(hm == 0)
      inds_bg = inds_bg_r * output_w + inds_bg_c

      if opt.dataset == 'kitti':
        valid_ind = torch.where(cls != -1) # known classes
      else:
        valid_ind = torch.where(cls == -1) # unknown classes
      ind = ind[valid_ind]
      cls = cls[valid_ind]

      y = torch.div(indices[0], output_w, rounding_mode='floor')
      x = indices[0] % output_w

      height = whs[0, :, 1]
      width = whs[0, :, 0]

      top = y - height / 2
      left = x - width / 2

      log_probs_B_Y, log_prob_bg, log_probs_all = gmm_forward(self.model, gaussians_models, results, indices[0], inds_bg, whs[0], opt.output_w, opt.output_h, opt.num_classes, opt.device, opt.flip_test)
      if log_probs_B_Y is not None:
        #density = uncertainty_confidence.logsumexp(log_probs_B_Y)
        #density_bg = uncertainty_confidence.logsumexp(log_prob_bg)

        densities_all = [
          uncertainty_confidence.logsumexp(log_prob_all)
          for log_prob_all in log_probs_all
        ]

        densities_bbox_list = [
          [
            crop(density_all.reshape(hm.shape), t, l, h, w) if (h > 0 and w > 0) else None for t, l, h, w in zip(top.int(), left.int(), height.int(), width.int())
          ] for density_all in densities_all
        ]

        """ heatmaps_bbox_list = [
          crop(hms[0], t, l, h, w) if (h > 0 and w > 0) else None for t, l, h, w in zip(top.int(), left.int(), height.int(), width.int())
        ] """

        preds_overdet = [] # 0 or 1 * 100

        """ for i in range(100):
          if densities_bbox_list[0][i] is not None:
            densities_image = []
            for j in range(3):
              densities_ch = densities_bbox_list[j][i]

              h, w = densities_ch.shape
              hp = int((WIDTH - w) / 2)
              vp = int((HEIGHT - h) / 2)
              padding_density = (hp, WIDTH - (w + hp), vp, HEIGHT - (h + vp))

              density_pad = F.pad(densities_ch, padding_density, 'constant', 0)

              density_pad = (density_pad - X_min[j]) / (X_max[j] - X_min[j])

              densities_image.append(density_pad)
            densities_image = torch.stack(densities_image, dim=0)
            pred = net(densities_image.unsqueeze(0).float().to(device))
            pred = torch.sigmoid(pred)
            preds_overdet.append(int(pred.item() >= 0.5))
          else:
            preds_overdet.append(0) """
        
        #densities_list = [uncertainty_confidence.logsumexp(log_probs) if log_probs is not None else None for log_probs in log_probs_list]
        #uncertainty = density - train_min_density
      else:
        density, uncertainty = None, None
      torch.cuda.synchronize()

      #gmm_results['hm'] = output['hm'].sigmoid_()
      #gmm_results['heatmaps_bbox_list'] = heatmaps_bbox_list
      #gmm_results['density'] = density
      #gmm_results['density_bg'] = density_bg
      #gmm_results['densities_all'] = densities_all
      gmm_results['densities_bbox_list'] = densities_bbox_list
      #gmm_results['preds_overdet'] = preds_overdet
      #gmm_results['e_uncertainty'] = uncertainty

      all_hm = output['hm'] # 1, 3, 96, 320
      r = torch.div(indices[0], opt.output_w, rounding_mode='floor')
      c = indices[0] % opt.output_w
      logits = all_hm[0, :, r, c].sigmoid_().permute(1, 0) # 100, 3
      entropy = uncertainty_confidence.entropy(logits)
      torch.cuda.synchronize()

      gmm_results['logits'] = logits
      #gmm_results['entropy'] = entropy

    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time

    if opt.debug >= 1:
      self.show_results(debugger, image, results, image_or_path_or_tensor)
        
    return {'results': results, 'ind': indices, 'gmm_results': gmm_results, 'uncs': uncertainties, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}