from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os


class BasePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self, net: nn.Module, data_loader: DataLoader):
        pred_list, conf_list, label_list = [], [], []
        for batch in data_loader:
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            pred, conf = self.postprocess(net, data)
            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                label_list.append(label[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        label_list = np.array(label_list, dtype=int)

        return pred_list, conf_list, label_list

    @torch.no_grad()
    def extract_stats(self, net, save_pth, ood_data_loader):
        net.eval()
        
        pred_list, _, label_list = self.inference(net, ood_data_loader)
        correct_preds = (pred_list == label_list)

        print('saving to ' + save_pth + ': {0} accuracy'.format(np.sum(correct_preds) / len(correct_preds)))
        preds_file_pth = os.path.join(save_pth, 'correct_preds.npy')
        np.save(preds_file_pth, correct_preds)