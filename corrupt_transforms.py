import numpy as np
import torch
from torchvision import transforms


class GaussianNoise(object):
    """Apply gaussian noise on PIL input

    Args:
        x (PIL Image): Image to be processed.
        
    Returns:
        x (Numpy Object): Processed image stored in numpy
    """
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def __call__(self, x):
        x = np.array(x) / 255.
        # must convert to uint8
        return np.uint8(np.clip(x + np.random.normal(size=x.shape, loc=self.mean, scale=self.std), 0, 1) * 255)

    def __repr__(self):
        return self.__class__.__name__ + 'mean={}, std={}'.format(self.mean, self.std)
    

def get_corruption(corruption, severity):
    if corruption == 'gaussian_noise':
        return GaussianNoise(mean=0, std=severity)
    else:
        raise RuntimeError('---> corruption: {} not implemented'.format(corruption))
    
