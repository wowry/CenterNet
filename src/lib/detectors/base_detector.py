from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import math
import torch
from torchvision.transforms.functional import crop
from models.model import create_model, load_model
from utils.image import get_affine_transform, gaussian_radius, draw_umich_gaussian2, gaussian2D
from utils.debugger import Debugger
from lib.utils.gmm_utils import gmm_forward
import utils.uncertainty_confidence as uncertainty_confidence
from lib.models.networks.UNet import UNet
from lib.models.networks.UNet_2inputs import UNet as UNet_2inputs
from lib.models.networks.UNet_2inputs2 import UNet as UNet_2inputs2
import matplotlib.pyplot as plt

IS_NN = 0
#prefix = 'threshold_center_'
#prefix = 'threshold_'
#prefix = '4ch_'
#prefix = '2inputs_'
prefix = '2inputs-2_rawhm_'

HEIGHT = 96
WIDTH = 96

THRESHOLD = 0.15

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

    gmm_dir = f'/work/shuhei-ky/exp/CenterNet/models/gmm/{opt.exp_id}'
    
    if 'threshold' in prefix:
      self.density_threshold = torch.load(f"{gmm_dir}/{prefix}density_threshold_{opt.arch}.pt")
    else:
      if '4ch' in prefix:
        self.overdet_model = UNet(4, 1, bilinear=False)
      elif prefix == '2inputs_':
        self.overdet_model = UNet_2inputs(3, 1, 1, bilinear=False)
      else:
        self.overdet_model = UNet_2inputs2(3, 1, 1, bilinear=False)
      self.overdet_model.load_state_dict(torch.load(f'{gmm_dir}/{prefix}u-net_weight.pt'))
      self.overdet_model = self.overdet_model.to(opt.device)
      self.overdet_model.eval()

      self.X_mean = []
      self.X_std = []
      for i in range(3):
        self.X_mean.append(torch.load(f'{gmm_dir}/X{i}_mean.pt'))
        self.X_std.append(torch.load(f'{gmm_dir}/X{i}_std.pt'))

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

  def run(self, image_or_path_or_tensor, gmm_dict=None, batch=None, meta=None, X2=None):
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
    classes = []
    raw_dets = []
    detections = []
    indices = []
    hms = []
    whs = []
    regs= []
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
      
      output, dets, inds, hm, wh, reg, uncs, forward_time = self.process(images, return_time=True)
      raw_dets.append(dets.clone())
      clses = dets[0, :, -1]
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
      classes.append(clses)
      indices.append(inds)
      whs.append(wh)
      regs.append(reg)
      uncertainties.append(uncs)
    
    results = self.merge_outputs(detections)
    output = outputs[0] # TODO: move into merge_outputs
    clses = classes[0] # TODO: move into merge_outputs
    indices = indices[0] # TODO: move into merge_outputs
    hms = hms[0] # TODO: move into merge_outputs
    whs = whs[0] # TODO: move into merge_outputs
    regs = regs[0]
    raw_dets = raw_dets[0]
    uncertainties = uncertainties[0] # TODO: move into merge_outputs

    torch.cuda.synchronize()
    
    gmm_results = {}
    if gmm_dict is not None:
      gaussians_models = gmm_dict['gaussians_models']

      raw_dets[0, :, 0] -= regs[0, :, 0]
      raw_dets[0, :, 1] -= regs[0, :, 1]

      # rawdetsをクラス順にソート
      classes = raw_dets[0, :, -1]
      ret = []
      for cls in range(3):
        inds = (classes == cls)
        ret.append(raw_dets[0, inds, :])
      ret = torch.cat(ret, dim=0)

      left = ret[:, 0]
      top = ret[:, 1]
      width = ret[:, 2] - ret[:, 0]
      height = ret[:, 3] - ret[:, 1]
      size = torch.stack([width, height], dim=1)
      score = ret[:, 4]
      center = torch.stack([left + width / 2, top + height / 2], dim=1) # 100 2
      x = center[:, 0]
      y = center[:, 1]

      log_probs_all = gmm_forward(self.model, gaussians_models, only_last=('threshold' in prefix))

      densities_all = [
        uncertainty_confidence.logsumexp(log_prob_all)
        for log_prob_all in log_probs_all
      ]
      densities_all = torch.stack(densities_all).reshape(len(densities_all), opt.output_h, opt.output_w)

      if prefix == 'threshold_center_':
        densities_center = densities_all[0].reshape(opt.output_h, opt.output_w)[y.long(), x.long()]

        preds_overdet = []

        for density in densities_center:
          if density is not None:
            preds_overdet.append(int(density < self.density_threshold))
          else:
            preds_overdet.append(0)
      elif prefix == 'threshold_':
        densities_layer_list = [crop(densities_all[0].reshape(opt.output_h, opt.output_w), t, l, HEIGHT, WIDTH) if (h > 0 and w > 0 and s >= THRESHOLD) else None for t, l, h, w, s in zip(top.int(), left.int(), height.int(), width.int(), score)]

        preds_overdet = []
        densities_mean = []

        for density in densities_layer_list:
          if density is not None:
            densities_mean.append(density.mean())
            preds_overdet.append(int(density.mean() < self.density_threshold))
          else:
            densities_mean.append(None)
            preds_overdet.append(0)
      else:
        X = densities_all
        for i in range(3):
          X[i] = (X[i] - self.X_mean[i]) / self.X_std[i]
        
        """ hm = np.zeros((opt.output_h, opt.output_w), dtype=np.float32)

        radius = [max(0, int(gaussian_radius((math.ceil(s[1].item()), math.ceil(s[0].item()))))) for s in size]
        ct_int = [c.int().numpy() for c in center.cpu()]
        diameter = [2 * r + 1 for r in radius]
        gaussian = [gaussian2D((d, d), sigma=d / 6) for d in diameter]
        hm_list = [draw_umich_gaussian2(hm, ct_int[i], radius[i], gaussian[i]) for i in range(len(center))] """
    
        """ for c, s in zip(center.cpu(), size):
          w, h = s
          radius = gaussian_radius((math.ceil(h.item()), math.ceil(w.item())))
          radius = max(0, int(radius))
          ct_int = c.int().numpy()
          diameter = 2 * radius + 1
          gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
          draw_umich_gaussian2(hm, ct_int, radius, gaussian) """
        
        X2 = torch.sum(hms[0], dim=0)

        X = X.unsqueeze(0).float()
        #X2 = torch.from_numpy(hm).clone().to(opt.device)
        X2 = X2.unsqueeze(0).unsqueeze(0).float()

        if '4ch' in prefix:
          images = torch.cat([X, X2], dim=1)
          output = self.overdet_model(images)
        elif '2inputs' in prefix:
          output = self.overdet_model(X, X2)
        output = output.squeeze()

        output = output[y.long(), x.long()]
        preds_overdet = (output.clone().detach() < 0.3).int()


      """ heatmaps_bbox_list = [
        crop(hms[0], t, l, HEIGHT, WIDTH) if (h > 0 and w > 0 and s >= THRESHOLD) else None for t, l, h, w, s in zip(top.int(), left.int(), height.int(), width.int(), score)
      ] """

      #preds_overdet = [] # 0 or 1 * 100

      if IS_NN == True:
        pass
        #pred_list = pred[y, x]

        """ X_mean = []
        X_std = []

        for i in range(3):
          X_mean.append(torch.load(f"/work/shuhei-ky/exp/CenterNet/models/gmm/bdd_centernet-2/X{i}_mean.pt"))
          X_std.append(torch.load(f"/work/shuhei-ky/exp/CenterNet/models/gmm/bdd_centernet-2/X{i}_std.pt")) """

        """ for i in range(100):
          if densities_bbox_list[0][i] is not None:
            cls = clses[i]
            densities_image = []
            for j in range(3):
              density = densities_bbox_list[j][i]
              heatmap = heatmaps_bbox_list[i]

              h, w = densities_ch.shape
              hp = int((WIDTH - w) / 2)
              vp = int((HEIGHT - h) / 2)
              padding_density = (hp, WIDTH - (w + hp), vp, HEIGHT - (h + vp))

              density_pad = F.pad(densities_ch, padding_density, 'constant', 0)

              density_pad = (density_pad - X_mean[j]) / X_std[j]

              densities_image.append(density_pad)
              density *= heatmap[cls.int().item()]
              density = (density - X_mean[j]) / X_std[j]
              densities_image.append(density)
            densities_image = torch.stack(densities_image, dim=0).unsqueeze(0).float().to(opt.device)
            pred = net(densities_image)
            pred = torch.sigmoid(pred)
            preds_overdet.append(int(pred.item() >= 0.5))
          else:
            preds_overdet.append(0) """

      #gmm_results['densities_bbox_list'] = densities_bbox_list
      #gmm_results['heatmaps_bbox_list'] = heatmaps_bbox_list
      #gmm_results['heatmaps_all'] = hms[0]
      #gmm_results['size'] = size
      #gmm_results['center'] = center
      gmm_results['preds_overdet'] = preds_overdet
      if prefix == 'threshold_':
        gmm_results['densities_mean'] = densities_mean
      #gmm_results['densities_center'] = densities_center
      #gmm_results['densities_all_0'] = densities_all[0].reshape(hm.shape[2:])
      #gmm_results['densities_all_1'] = densities_all[1].reshape(hm.shape[2:])
      #gmm_results['densities_all_2'] = densities_all[2].reshape(hm.shape[2:])

    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time

    if opt.debug >= 1:
      self.show_results(debugger, image, results, image_or_path_or_tensor)
        
    return {'results': results, 'ind': indices, 'gmm_results': gmm_results, 'uncs': uncertainties, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}