import pathlib
import re
import os
import numpy as np

def get_file_index_str(img_idx):
    return "{:06d}".format(img_idx)

def get_unc_file(filename):
  uncertainties = {}
  uncertainties.update({
      'objectness': [],
      'location': [],
      'dimensions': [],
      'class': []
  })
  with open(filename, 'r') as f:
    lines = f.readlines()
  content = [line.strip().split(' ') for line in lines]
  uncertainties['objectness'] = np.array([float(x[0]) for x in content])
  uncertainties['location'] = np.array([[float(x[1]), float(x[2])] for x in content]) # x, y
  uncertainties['dimensions'] = np.array([[float(x[3]), float(x[4])] for x in content]) # w, h
  uncertainties['class'] = np.array([[float(cu) for cu in x[5:]] for x in content])
  return uncertainties

def get_unc_file2(filename):
  u_obj = {}
  u_obj.update({
      'gt': [],
      'pred': [],
  })
  with open(filename, 'r') as f:
    lines = f.readlines()
  contents = [line.strip().split(' ') for line in lines]
  for content in contents:
    cls_id = content[0]
    hm_preds = [float(x) for x in content[1:]]
    hm_gts = [0, 0, 0]
    hm_gts[int(cls_id)] = 1

    u_obj['gt'].append(hm_gts)
    u_obj['pred'].append(hm_preds)

  return u_obj

def get_u_obj_files(filename):
  u_obj = {'true': [], 'pred': []}
  with open(filename, 'r') as f:
    lines = f.readlines()
  contents = [line.strip().split(' ') for line in lines]
  for content in contents:
    cls_id = int(float(content[0]))
    hm_pred = [float(x) for x in content[1:]]
    num_cls = len(hm_pred)
    hm_true = [1] * num_cls
    hm_true[cls_id] = 0

    u_obj['true'].append(hm_true)
    u_obj['pred'].append(hm_pred)

  return u_obj

def get_unc_files(uncs_folder_path):
  obj_keys = ['obj', 'loc', 'dim', 'cls']
  uncs = {}
  for key in obj_keys:
    unc_folder_path = os.path.join(uncs_folder_path, key)
    filepaths = pathlib.Path(unc_folder_path).glob('*.txt')
    prog = re.compile(r'^\d{6}.txt$')
    filepaths = filter(lambda f: prog.match(f.name), filepaths)
    image_ids = [int(p.stem) for p in filepaths]
    image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
      image_ids = list(range(image_ids))
    unc = []
    unc_folder_path = pathlib.Path(unc_folder_path)
    for i, idx in enumerate(image_ids):
      image_idx = get_file_index_str(idx)
      filename = unc_folder_path / (image_idx + '.txt')
      if key == 'obj':
        unc.append(get_u_obj_files(filename))
    uncs[key] = unc
  return uncs

def eval_encs2(image_ids, uncs):
  pred = np.array([u['pre'] for u in uncs])
  gt = np.array([u['gt'] for u in uncs])

def eval_encs(image_ids, uncs):
  U_obj = np.array([u['objectness'] for u in uncs])[..., np.newaxis]
  U_loc = np.array([u['location'] for u in uncs])
  U_dim = np.array([u['dimensions'] for u in uncs])
  U_cls = np.array([u['class'] for u in uncs])

  U = np.concatenate([U_obj, U_loc, U_dim, U_cls], axis=2) # 4, 100, 8

  thre = 0.7

  idx_list = np.array(list(zip(*np.where(
    (U[..., 0] < (1 - thre))
    & (
      (U[..., 1] > thre)
      | (U[..., 2] > thre)
      | (U[..., 3] > thre)
      | (U[..., 4] > thre)
    )
  ))))
  
  U_obj_sorted = np.sort(U_obj)[::-1]
  U_obj_sorted_idx = np.argsort(U_obj)[::-1]

  U_loc_x = U_loc[..., 0]
  thre = 0.7
  print('U_loc =================')
  print(np.array(list(zip(*np.where(U_loc > thre)))))
  print('U_dim =================')
  print(np.array(list(zip(*np.where(U_dim > thre)))))

  idx_list = np.array(list(zip(*np.where(U_dim > thre))))
  print(image_ids[idx_list[:, 0]])
