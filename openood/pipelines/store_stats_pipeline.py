import time
import os

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger


class StoreStatsPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)

        # init network
        net = get_network(self.config.network)

        # init ood postprocessor
        postprocessor = get_postprocessor(self.config)
        # setup for distance-based methods
        postprocessor.setup(net, id_loader_dict, ood_loader_dict)
        print('\n', flush=True)
        print(u'\u2500' * 70, flush=True)

        # start evaluating ood detection methods
        for split in self.config.ood_dataset.split_names:
            for dataset in ood_loader_dict[split]:
                save_pth = os.path.join(self.config.stats_dir, self.config.network.name, dataset, self.config.postprocessor.name)

                if not os.path.exists(save_pth):
                    os.makedirs(save_pth)

                timer = time.time()
                postprocessor.extract_stats(net, save_pth, ood_loader_dict[split][dataset])

                if self.config.postprocessor.name == 'gmm_ensemble':
                    break # for gmm_ensemble each call to extract_stats compute the score for all ood datasets so we should stop the loop

                print('Time used to compute statistics on ' + dataset + ' : {:.0f}s'.format(time.time() - timer))
