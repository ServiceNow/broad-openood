"""Adapted from: https://github.com/facebookresearch/odin."""
from typing import Any

import torch
import torch.nn as nn
import numpy as np
import os

from .base_postprocessor import BasePostprocessor


class ODINPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

        self.temperature = self.args.temperature
        self.noise = self.args.noise
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def postprocess(self, net: nn.Module, data: Any):
        data.requires_grad = True
        output = net(data)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        criterion = nn.CrossEntropyLoss()

        labels = output.detach().argmax(axis=1)

        # Using temperature scaling
        output = output / self.temperature

        loss = criterion(output, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2

        # Scaling values taken from original code
        gradient[:, 0] = (gradient[:, 0]) / (63.0 / 255.0)
        gradient[:, 1] = (gradient[:, 1]) / (62.1 / 255.0)
        gradient[:, 2] = (gradient[:, 2]) / (66.7 / 255.0)

        # Adding small perturbations to images
        tempInputs = torch.add(data.detach(), gradient, alpha=-self.noise)
        output = net(tempInputs)
        output = output / self.temperature

        # Calculating the confidence after adding perturbations
        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

        conf, pred = nnOutput.max(dim=1)

        return pred, conf

    def extract_stats(self, net, save_pth, ood_data_loader):
        net.eval()
    
        odin = []
        for batch in ood_data_loader:
            batch = batch['data'].cuda()
            _, scores = self.postprocess(net, batch)
            odin.extend(scores.cpu())
        odin = np.array(odin)

        file_pth = os.path.join(save_pth, 'odin.npy')
        np.save(file_pth, odin)

    def set_hyperparam(self, hyperparam: list):
        self.temperature = hyperparam[0]
        self.noise = hyperparam[1]

    def get_hyperparam(self):
        return [self.temperature, self.noise]
