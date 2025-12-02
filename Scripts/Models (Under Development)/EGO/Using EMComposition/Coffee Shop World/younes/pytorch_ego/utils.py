import random

import numpy as np
import torch


def set_random_seed(params):
    try:
        seed = params.seed
    except:
        seed = params
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
