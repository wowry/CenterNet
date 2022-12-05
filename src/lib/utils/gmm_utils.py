import math
import torch
from torch import nn
from tqdm import tqdm
from models.utils import flip_tensor

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def get_embeddings(
    net, loader: torch.utils.data.DataLoader, output_w, device, flip_test=False
):
    embeddings = torch.empty(0, device=device)
    labels = torch.empty(0, device=device)

    with torch.no_grad():
        for batch in tqdm(loader):
            data = batch['input'].to(device)
            ind = batch['ind'][0].to(device)
            cls = batch['cls'][0].to(device)

            valid_ind = torch.where(cls != -1)
            ind = ind[valid_ind]
            cls = cls[valid_ind]

            data = data.to(device)

            if isinstance(net, nn.DataParallel):
                out = net.module(data)
                out = net.module.feature
            else:
                out = net(data)
                out = net.feature # 1, 64, 96, 320
                        
            r = torch.div(ind, output_w, rounding_mode='floor')
            c = ind % output_w

            feature = out[0, :, r, c].permute(1, 0) # 64, N

            embeddings = torch.cat((embeddings, feature), dim=0).double()
            labels = torch.cat((labels, cls), dim=0)
    
    print(embeddings.size())
    print(labels.size())

    return embeddings, labels


def gmm_forward(net, gaussians_model, ind, inds_bg, output_w, num_classes, device, flip_test=False):
    log_probs_B_Y, log_prob_bg = None, None

    if isinstance(net, nn.DataParallel):
        out = net.module.feature
    else:
        out = net.feature
    
    r = torch.div(ind, output_w, rounding_mode='floor')
    c = ind % output_w

    r_bg = torch.div(inds_bg, output_w, rounding_mode='floor')
    c_bg = inds_bg % output_w

    features_B_Z = out[0, :, r, c].permute(1, 0) # 64, N
    #features_B_Z = out[0, :, :, :].reshape(out.size()[1], -1).permute(1, 0) # 64, N

    features_bg = out[0, :, r_bg, c_bg].permute(1, 0) # 64, N

    if features_B_Z.size()[0] > 0:
        log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :].double()) # N, 3
        log_prob_bg = gaussians_model.log_prob(features_bg[:, None, :].double()) # N, 3

    return log_probs_B_Y, log_prob_bg


def gmm_evaluate(net, gaussians_model, loader, output_w, output_h, num_classes, device):
    logits_N_C = torch.empty(0, device=device)
    logits_bg = torch.empty(0, device=device)

    with torch.no_grad():
        for batch in tqdm(loader):
            data = batch['input'].to(device)
            hm = batch['hm'].to(device) # 1, 3, 96, 320
            ind = batch['ind'][0].to(device)
            cls = batch['cls'][0].to(device)

            hm = torch.sum(hm[0, :, :, :], dim=0)
            inds_bg_r, inds_bg_c = torch.where(hm == 0)
            inds_bg = inds_bg_r * output_w + inds_bg_c

            """ all_inds = torch.arange(0, output_w * output_h, device=device).reshape(output_h, output_w)
            obj_inds = torch.empty(0, device=device)
            for i, idx in enumerate(ind):
                width, height = wh[i]
                left, right = (idx - (width / 2)) % output_w, (idx + (width / 2)) % output_w
                top, bottom = (idx - (height / 2) * output_w) // output_w, (idx + (height / 2) * output_w) // output_w
                target_r, target_c = torch.where((all_inds % output_w >= left) & (all_inds % output_w <= right) & (all_inds // output_w >= top) & (all_inds // output_w <= bottom))
                target_inds = target_r * output_w + target_c
                obj_inds = torch.cat((obj_inds, target_inds), dim=0)
            bg_inds = torch.tensor([i for i in all_inds.reshape(-1) if i not in obj_inds], device=device) """

            valid_ind = torch.where(cls != -1)
            ind = ind[valid_ind]
            cls = cls[valid_ind]

            data = data.to(device)

            if isinstance(net, nn.DataParallel):
                out = net.module(data)
            else:
                out = net(data)
            
            logit_B_C, log_prob_bg = gmm_forward(net, gaussians_model, ind, inds_bg, output_w, num_classes, device)

            if logit_B_C is not None:
                logits_N_C = torch.cat((logits_N_C, logit_B_C), dim=0)
                logits_bg = torch.cat((logits_bg, log_prob_bg), dim=0)

    return logits_N_C, logits_bg


def gmm_get_logits(gmm, embeddings):

    log_probs_B_Y = gmm.log_prob(embeddings[:, None, :])
    return log_probs_B_Y


def gmm_fit(embeddings, labels, num_classes):
    with torch.no_grad():
        classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)])
        classwise_cov_features = torch.stack(
            [centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c]) for c in range(num_classes)]
        )

    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1], device=classwise_cov_features.device,
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter),
                )
            except:
                continue
            break

    return gmm, jitter_eps
