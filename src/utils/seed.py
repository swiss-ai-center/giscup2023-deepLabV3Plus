import random

import numpy as np
import torch


def set_seed(seed):
    """Helper function to fix the seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
