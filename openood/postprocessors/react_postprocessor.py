from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os

from .base_postprocessor import BasePostprocessor


class ReactPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(ReactPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        activation_log = []
        net.eval()
        with torch.no_grad():
            for batch in tqdm(id_loader_dict['train'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()

                _, feature = net(data, return_feature=True)

                activation_log.append(feature.data.cpu().numpy())

        activation_log = np.concatenate(activation_log, axis=0)
        self.threshold = np.percentile(activation_log.flatten(),
                                       self.percentile)
        print('Threshold at percentile {:2d} over id data is: {}'.format(
            self.percentile, self.threshold))

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net.forward_threshold(data, self.threshold)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf
   
    @torch.no_grad()
    def extract_stats(self, net, save_pth, ood_data_loader):
        net.eval()
    
        react_scores = []
        for batch in ood_data_loader:
            batch = batch['data'].cuda()
            _, score = self.postprocess(net, batch)
            react_scores.extend(score.cpu().tolist())
        react_scores = np.array(react_scores)

        react_file_pth = os.path.join(save_pth, 'react.npy')
        np.save(react_file_pth, react_scores)

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]

    def get_hyperparam(self):
        return self.percentile
