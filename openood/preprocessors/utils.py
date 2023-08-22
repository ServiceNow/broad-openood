from openood.utils import Config

from .base_preprocessor import BasePreprocessor
from .cutpaste_preprocessor import CutPastePreprocessor
from .draem_preprocessor import DRAEMPreprocessor
from .pixmix_preprocessor import PixMixPreprocessor
from .test_preprocessor import TestStandardPreProcessor
from .cadet_preprocessor import CadetPreprocessor


def get_preprocessor(config: Config, split):
    train_preprocessors = {
        'base': BasePreprocessor,
        'draem': DRAEMPreprocessor,
        'cutpaste': CutPastePreprocessor,
        'pixmix': PixMixPreprocessor,
        'cadet': CadetPreprocessor,
    }
    test_preprocessors = {
        'base': TestStandardPreProcessor,
        'draem': DRAEMPreprocessor,
        'cutpaste': CutPastePreprocessor,
        'cadet': CadetPreprocessor
    }

    if split == 'train':
        return train_preprocessors[config.preprocessor.name](config)
    else:
        return test_preprocessors[config.preprocessor.name](config)
