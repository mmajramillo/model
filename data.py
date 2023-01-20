# %% 0. Import dependencies
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray as Array
from sklearn.datasets import make_blobs, make_moons, make_circles
from functools import partial
from typing import Callable, Dict
from enum import Enum

# %% 1. Define an abstract parent class
class AbstractDataset(ABC):
    n_samples: int                         # Defien the attributes
    seed : int
    noise: float
    f: callable
    
    def __init__(self, n_samples:int, noise:float, seed:int=0):
        self.n_samples = n_samples
        self.noise = noise
        self.seed = seed
    
    @abstractmethod
    def init(self) -> None:
        ...
    
    def sample(self) -> Array:
        X, _ = self.f()
        return X 

# %% 2. Define several child classes to generate datasets
#2.1 Blobs
class BlobsDataset(AbstractDataset):
    def __init__(self, *args, **kwargs):
        super(BlobsDataset, self).__init__(*args, **kwargs)
        
    def init(self):
        self.f = partial(
                         make_blobs,
                         n_samples = self.n_samples,
                         random_state = self.seed,
                         cluster_std = self.noise
        )

# 2.2 Copile steps 2.2 and 2.3 into one single class
class BiClusterEnum(Enum):
    CIRCLES = "CIRCLES"              # Set the options
    MOONS = "MOONS"

class BiClusterDataset(AbstractDataset):
    
    dataset_f: Dict(BiClusterEnum, Callable) = {
        BiClusterEnum.CIRCLES : make_circles,
        BiClusterEnum.MOONS : make_moons
    }
    def __init__(self, dataset_type: BiClusterEnum, *args, **kwargs):
        super(BiClusterDataset, self).__init__(*args, **kwargs)
        self.data_f = self.dataset_f[dataset_type]
        
    def init(self):
        self.f = partial(
                         self.data_f,
                         n_samples = self.n_samples,
                         noise = self.noise,
                         random_state = self.seed,
        )

#2.2 Moons
class MoonsDataset(AbstractDataset):
    def __init__(self, *args, **kwargs):
        super(MoonsDataset, self).__init__(*args, **kwargs)
        
    def init(self):
        self.f = partial(
                         make_moons,
                         n_samples = self.n_samples,
                         noise = self.noise,
                         random_state = self.seed,
        )
#2.3 Circles        
class CirclesDataset(AbstractDataset):
    def __init__(self, *args, **kwargs):
        super(CirclesDataset, self).__init__(*args, **kwargs)
        
    def init(self):
        self.f = partial(
                         make_circles,
                         n_samples = self.n_samples,
                         noise = self.noise,
                         random_state = self.seed,
        )

#2.4 test

BiClusterEnum.CIRCLES


ds = CirclesDataset(10,0.1)
ds = BlobsDataset(10,0.1)
ds = MoonsDataset(10,0.1)

BiClusterDataset(BiClusterEnum.CIRCLES, 100, 0.1)

ds.init()
ds.sample()


# %%
