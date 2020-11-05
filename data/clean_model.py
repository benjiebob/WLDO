import sys
sys.path.append("../")

import torch
from global_utils import config
import pickle as pkl

delete_keys = [
    "module.netG_DETAIL.texture_predictor",
    "module.netG_DETAIL.code_predictor.shape_predictor",
    "module.netG_DETAIL.mean_v",
    "module.netG_DETAIL.vert2kp"
]

model = torch.load('../data/pretrained/3501_00034.pth')


delete_list = []
for params in model.keys():
    print (params)
    for search_key in delete_keys:
        if search_key in params:
            delete_list.append(params)

model_clean = model.copy()
for s in delete_list:
    del model_clean[s]

betas_bias = 'module.netG_DETAIL.code_predictor.betas_predictor.pred_layer.bias'
betas_scale_bias = 'module.netG_DETAIL.code_predictor.betas_scale_predictor.pred_layer.bias'
betas_scale_weight = 'module.netG_DETAIL.code_predictor.betas_scale_predictor.pred_layer.weight'

with open(config.SMAL_DATA_FILE, 'rb') as f:
    u = pkl._Unpickler(f)
    u.encoding = 'latin1'
    data = u.load()

# Select mean shape for quadruped type
shape_cluster_means = torch.from_numpy(data['cluster_means'][1][None, :20]).float()
shape_cluster_means_torch = shape_cluster_means.to(model_clean[betas_bias].device)

model_clean[betas_bias] = model_clean[betas_bias] - shape_cluster_means_torch[0]
model_clean[betas_scale_bias] = model_clean[betas_scale_bias][:6]
model_clean[betas_scale_weight] = model_clean[betas_scale_weight][:6, :]

torch.save(model_clean, '../data/pretrained/3501_00034_betas_v2.pth')
print ("Done")
