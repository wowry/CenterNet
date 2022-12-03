from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
import matplotlib.pyplot as plt
import numpy as np
import wandb


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  wandb.init(
    project=f'{opt.dataset}_{opt.task}',
    name=f'{opt.exp_id}_train',
    group=opt.exp_id,
    config=opt)
  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10

  train_log_list = []
  val_log_list = []

  # epochs of increasing momentum
  m_epochs = [5, 20, 60]

  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    if 'certainnet' in opt.arch and opt.ablation >= 4:
      if epoch in m_epochs:
        if epoch == m_epochs[0]:
          gamma = 0.99
        elif epoch == m_epochs[1]:
          gamma = 0.999
        elif epoch == m_epochs[2]:
          gamma = 0.9999      
        opt.gamma = gamma
        print('Increase momentum to', opt.gamma)

    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    train_log_list.append(log_dict_train)
    logger.write('epoch: {} |'.format(epoch))

    wandb_logs = {'epoch': epoch}
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
      wandb_logs['train_{}'.format(k)] = v
    wandb.log(wandb_logs)
    wandb.watch(model)
    
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
        val_log_list.append(log_dict_val)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
      wandb.save("model_last.pth")
    logger.write('\n')
    if (epoch in opt.lr_step) or (epoch % 10 == 0):
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
    if epoch in opt.lr_step:
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

  if len(train_log_list) > 0:
    x_train = np.arange(1, opt.num_epochs + 1)
    for k in train_log_list[0]:
      if 'time' in k:
        continue
      v = [d[k] for d in train_log_list]
      plt.figure()
      plt.plot(x_train, v)
      plt.savefig(os.path.join(opt.save_dir, f"train-{k}.png"))
  if len(val_log_list) > 0:
    x_val = np.arange(5, opt.num_epochs + 1, 5)
    for k in val_log_list[0]:
      if 'time' in k:
        continue
      v = [d[k] for d in val_log_list]
      plt.figure()
      plt.plot(x_val, v)
      plt.savefig(os.path.join(opt.save_dir, f"val-{k}.png"))

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)