from typing import Any
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os


class GMMEnsemblePostprocessor:
    def __init__(self, config):
        self.config = config
        self.stats = config.postprocessor.postprocessor_args.stats
        self.stats_pth = config.stats_dir
        self.n_components = config.postprocessor.postprocessor_args.n_components

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        scores = []
        for stat_name in self.stats:
            score = self.get_score(self.config.postprocessor.postprocessor_args.id_ds_name, stat_name) # To train only on correct preds, save their scores apart
            scores.append(score)
        scores = np.array(scores)
        self.gm = GaussianMixture(n_components=self.n_components).fit(X=np.transpose(scores))

    @torch.no_grad()
    def extract_stats(self, net, save_pth, ood_data_loader):
        for dataset in self.config.ood_dataset.ood.datasets:
            save_pth = os.path.join(self.stats_pth, self.config.network.name, dataset, self.config.postprocessor.name)
            if not os.path.exists(save_pth):
                    os.makedirs(save_pth)
            
            ood_scores = []
            for stat_name in self.stats:
                score = self.get_score(dataset, stat_name)
                ood_scores.append(score)
            ood_scores = np.array(ood_scores)
            gm_log_llh = self.gm.score_samples(X=np.transpose(ood_scores))
            save_file_pth = os.path.join(save_pth, 'ens_scores.npy')
            
            np.save(save_file_pth, gm_log_llh)

    def get_score(self, dataset_name, stat_name):
        if stat_name == 'mds-all':
            if self.config.network.name == 'resnet50':
                layers = range(5)
            elif self.config.network.name == 'vit':
                layers = range(12)
            else:
                raise NotImplementedError
            scores = sum([np.load(os.path.join(self.stats_pth, self.config.network.name, dataset_name, 'mds',
                                               f'layer_{i}.npy')) for i in layers])
        elif stat_name == 'mds-last':
            if self.config.network.name == 'resnet50':
                last_layer = 4
            elif self.config.network.name == 'vit':
                last_layer = 11
            else:
                raise NotImplementedError
            scores = np.load(os.path.join(self.stats_pth, self.config.network.name, dataset_name, 'mds', f'layer_{last_layer}.npy'))
        else:
            score_pth = os.path.join(self.stats_pth, self.config.network.name, dataset_name, stat_name)
            scores = np.load(score_pth)
        return scores