import random
import numpy as np
import pandas as pd
import torch

def torch_fix_seed(seed= 42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def second2date(seconds: int) -> str:
    seconds = int(seconds)
    d = seconds // (3600*24)
    h = (seconds - 3600*24*d) // 3600
    m = (seconds - 3600*24*d - 3600*h) // 60
    s = seconds - 3600*24*d - 3600*h - 60*m

    return f'{d:0>2}:{h:0>2}:{m:0>2}:{s:0>2}'

def list2pdData(loss_list: list, metrics: list) -> pd.DataFrame:
    loss_dict = {}
    for key, values in zip(metrics, loss_list):
        loss_dict[key] = values
    return pd.DataFrame(loss_dict)
