import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    Resized,
    Spacingd,
    ThresholdIntensityd,
    NormalizeIntensityd,
    RandAffined
)
import random
from typing import List
from utils.train_utils import random_mask_patches_3d


from time import time

class TaskDataset(Dataset):
    """
    Dataset class for the lung segmentation

    Args:
        csv_file (str): path to the csv file containing the image and label paths
        mode (str): train, val or test
        mask (bool): whether to apply mask to the image
        mask_dir (str): path to the mask directory
        mask_size (List[int]): list of mask sizes
        mask_percent (List[int]): list of mask percentages
        seed (int): random seed
        offset (bool): whether to apply offset to the mask

    Returns:
        y1_img (torch.Tensor): image tensor
        y1_seg (torch.Tensor): reference segmentation tensor
    """
    def __init__(self,  
                 csv_file: str, 
                 mode: str = "train", 
                 mask: bool = False, 
                 mask_dir: str = None,
                 mask_size: List[int] = [7],
                 mask_percent: List[int] = [70],
                 seed: int = 0,
                 offset: bool = False,
                ) -> None:
        self.csv_path = pd.read_csv(csv_file)
        #train val split
        np.random.seed(seed)
        self.mode = mode
        self.mask = mask
        self.mask_dir = mask_dir
        self.mask_size = mask_size
        self.mask_percent = mask_percent
        self.offset = offset
    
        self.test = self.csv_path[self.csv_path['train'] == 0].reset_index(drop=True)#.iloc[0:2]
        self.train_raw = self.csv_path[self.csv_path['train'] == 1].reset_index(drop=True)

        self.train = self.train_raw.sample(frac=0.8, random_state=seed)
        self.val = self.train_raw.drop(self.train.index)
        self.train = self.train.reset_index(drop=True)#.iloc[0:2]
        self.val = self.val.reset_index(drop=True)#.iloc[0:2]
        
        print(f"Train: {len(self.train)} Val: {len(self.val)} Test: {len(self.test)}")

        if mode == "train":
            self.data = self.train
        elif mode == "val":
            self.data = self.val
        elif mode == "test":
            self.data = self.test
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.transforms_rand = Compose([
            LoadImaged(keys=['img', 'label']),
            EnsureChannelFirstd(keys=['img', 'label']),
            Orientationd(keys=['img', 'label'], axcodes='RAS'),
            Spacingd(keys=['img', 'label'], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            ThresholdIntensityd(keys = 'img', threshold =-1024.0, above = True, cval = -1024.0),
            ThresholdIntensityd(keys = 'img', threshold = 276.0, above = False, cval = 276.0,),
            NormalizeIntensityd(keys = 'img', subtrahend  = -370.00039267657144, divisor = 436.5998675471528),
            #ResizeWithPadOrCropd(keys=['img', 'label'], spatial_size =(224,224,224), mode = 'constant'),
            Resized(keys=['img', 'label'], spatial_size =(224,224,224)),
            RandAffined(
                keys=["img", "label"],
                mode=("bilinear", "nearest"),
                prob=1,
                spatial_size=(224,224,224),
                translate_range=(10, 10, 10),
                rotate_range=( np.pi / 36, np.pi / 36, np.pi / 36),
                scale_range=(0.001, 0.001, 0.001),
                padding_mode="border",
            )
        ])
        self.transforms = Compose([
            LoadImaged(keys=['img', 'label']),
            EnsureChannelFirstd(keys=['img', 'label']),
            Orientationd(keys=['img', 'label'], axcodes='RAS'),
            Spacingd(keys=['img', 'label'], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            ThresholdIntensityd(keys = 'img', threshold =-1024.0, above = True, cval = -1024.0),
            ThresholdIntensityd(keys = 'img', threshold = 276.0, above = False, cval = 276.0,),
            NormalizeIntensityd(keys = 'img', subtrahend  = -370.00039267657144, divisor = 436.5998675471528),
            Resized(keys=['img', 'label'], spatial_size =(224,224,224)),

            #ResizeWithPadOrCropd(keys=['img', 'label'], spatial_size =(224,224,224), mode = 'constant'),
        ])


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        #load image data
        y1_img = self.data.iloc[idx].y1_img#.replace('projects/longitudinal_lung','dataset')
        y1_seg = self.data.iloc[idx].y1_seg#.replace('projects/longitudinal_lung','dataset')

        #apply random transforms to the image in training mode
        #apply deterministic transforms to the image in val and test mode
        if self.mode == 'train':
            output = self.transforms_rand({'img': y1_img, 'label': y1_seg})
            y1_img, y1_seg = output['img'], output['label']
        else:
            output = self.transforms({'img': y1_img, 'label': y1_seg})
            y1_img, y1_seg = output['img'], output['label']

        #apply mask to the image            
        if self.mask:
            set_y1_img = torch.quantile(y1_img, 0.9)
            set_y1_img = 1.0
            ms = random.choice(self.mask_size)
            mp = random.choice(self.mask_percent)
            y1_img = random_mask_patches_3d(y1_img, 
                                                 patch_size=(ms, ms, ms), 
                                                 mask_percentage= mp, 
                                                 replace = set_y1_img,
                                                 offset = self.offset)

        return  y1_img, y1_seg

