import torchvision.transforms as tvs_trans

from torchvision import transforms
from openood.utils.config import Config
import torch

from .transform import Convert, interpolation_modes, normalization_dict


class CadetPreprocessor():
    """For train dataset standard transformation."""
    def __init__(self, config: Config):
        self.pre_size = config.dataset.pre_size
        self.image_size = config.dataset.image_size
        self.n_transforms = config.preprocessor.n_transforms
        self.crop_scale = config.preprocessor.crop_scale
        self.interpolation = interpolation_modes[config.dataset.interpolation]
        normalization_type = config.dataset.normalization_type
        if normalization_type in normalization_dict.keys():
            self.mean = normalization_dict[normalization_type][0]
            self.std = normalization_dict[normalization_type][1]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]

        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(self.pre_size, interpolation=self.interpolation),
            transforms.RandomResizedCrop(size=self.image_size, scale=(self.crop_scale[0], self.crop_scale[1])),
            tvs_trans.RandomHorizontalFlip(),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=self.mean, std=self.std),
        ])

    def setup(self, **kwargs):
        pass

    def __call__(self, image):
        return torch.stack([self.transform(image) for _ in range(self.n_transforms)])
