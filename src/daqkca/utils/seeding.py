import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.use_deterministic_algorithms(True, warn_only=True)
