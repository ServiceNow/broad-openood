from typing import Any

import numpy as np
import torch
import torch.nn as nn
import os

from .base_postprocessor import BasePostprocessor


class DoctorPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        
        output = net(data)
        softmax = torch.softmax(output, dim=1)
        g = torch.sum(torch.square(softmax), 1)
        score_alpha = (1-g)/g

        return score_alpha

    @torch.no_grad()
    def extract_stats(self, net, save_pth, ood_data_loader):
        net.eval()
        
        alpha_score = []
        for batch in ood_data_loader:
            batch = batch['data'].cuda()
            alpha= self.postprocess(net, batch)

            for idx in range(len(batch)):
                alpha_score.append(alpha[idx].cpu().tolist())
        alpha_score = np.array(alpha_score)

        alpha_file_pth = os.path.join(save_pth, 'alpha.npy')
        np.save(alpha_file_pth, alpha_score)

