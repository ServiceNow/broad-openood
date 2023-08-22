from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os

from .base_postprocessor import BasePostprocessor


class MaxLogitPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        conf, pred = torch.max(output, dim=1)
        return pred, conf

    @torch.no_grad()
    def extract_stats(self, net, save_pth, ood_data_loader):
        net.eval()
        
        maxlogit_list, logitnorm_list, softmax_list = [], [], []
        for batch in ood_data_loader:
            data = batch['data'].cuda()
            output = net(data)
            maxlogit, _ = torch.max(output, dim=1)
            logitnorm = torch.norm(output, dim=1)
            softmax, _ = torch.max(torch.softmax(output, dim=1), dim=1)
            
            for idx in range(len(data)):
                maxlogit_list.append(maxlogit[idx].cpu().tolist())
                logitnorm_list.append(logitnorm[idx].cpu().tolist())
                softmax_list.append(softmax[idx].cpu().tolist())

        # convert values into numpy array
        maxlogit_list = np.array(maxlogit_list)
        logitnorm_list = np.array(logitnorm_list)
        softmax_list = np.array(softmax_list)

        maxlogit_pth = os.path.join(save_pth, 'maxlogit.npy')
        np.save(maxlogit_pth, maxlogit_list)
        logitnorm_pth = os.path.join(save_pth, 'logitnorm.npy')
        np.save(logitnorm_pth, logitnorm_list)
        softmax_pth = os.path.join(save_pth, 'softmax.npy')
        np.save(softmax_pth, softmax_list)