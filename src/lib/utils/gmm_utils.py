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

            embeddings = torch.cat((embeddings, feature), dim=0)
            labels = torch.cat((labels, cls), dim=0)
    
    print(embeddings.size())
    print(labels.size())

    return embeddings, labels


def gmm_forward(net, gaussians_model, ind, output_w, num_classes, device, flip_test=False):
    log_probs_B_Y = None

    if isinstance(net, nn.DataParallel):
        out = net.module.feature
    else:
        out = net.feature
    
    r = torch.div(ind, output_w, rounding_mode='floor')
    c = ind % output_w

    features_B_Z = out[0, :, r, c].permute(1, 0) # 64, N
    #features_B_Z = out[0, :, :, :].reshape(out.size()[1], -1).permute(1, 0) # 64, N

    if features_B_Z.size()[0] > 0:
        log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :]) # N, 3

    return log_probs_B_Y


def gmm_evaluate(net, gaussians_model, loader, output_w, num_classes, device):
    logits_N_C = torch.empty(0, device=device)

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
            else:
                out = net(data)
            
            logit_B_C = gmm_forward(net, gaussians_model, ind, output_w, num_classes, device)

            if logit_B_C is not None:
                logits_N_C = torch.cat((logits_N_C, logit_B_C), dim=0)

    return logits_N_C


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
                print(gmm)
            except:
                continue
            break

    return gmm, jitter_eps
