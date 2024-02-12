import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np
from typing import Tuple


class TaskDataset(Dataset):
    def __init__(self, data_dir: str, csv_file: str, mode: str = "train") -> None:
        pass

    def __len__(self) -> int:
        return

    def __getitem__(self, idx: int):
        return
