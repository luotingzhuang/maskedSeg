import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np
from typing import Tuple
import pandas as pd
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    CenterSpatialCropd,
    Orientationd,
    ResizeWithPadOrCropd,
    Spacingd,
    FillHolesd,
    ScaleIntensityRanged,
    ThresholdIntensityd,
    NormalizeIntensityd,
    RandAffined
)


class TaskDataset(Dataset):
    def __init__(self,  
                 csv_file: str, 
                 mode: str = "train", 
                 mask: bool = False, 
                 mask_size: int = 20,
                 mask_percent: int = 40,
                 seed: int = 0) -> None:
        self.csv_path = pd.read_csv(csv_file)#.dropna()
        #train val split
        np.random.seed(seed)
        self.mode = mode
        self.mask = mask
        self.mask_size = mask_size
        self.mask_percent = mask_percent

        self.test = self.csv_path[self.csv_path['train'] == 0].reset_index(drop=True)#.iloc[0:2]
        self.train_raw = self.csv_path[self.csv_path['train'] == 1].reset_index(drop=True)

        self.train = self.train_raw.sample(frac=0.7, random_state=seed)
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
            ResizeWithPadOrCropd(keys=['img', 'label'], spatial_size =(224,224,224), mode = 'constant'),
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
            ResizeWithPadOrCropd(keys=['img', 'label'], spatial_size =(224,224,224), mode = 'constant'),
        ])


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        #load image data

        y0_img = self.data.iloc[idx].y0_img
        y0_seg = self.data.iloc[idx].y0_seg

        y1_img = self.data.iloc[idx].y1_img
        y1_seg = self.data.iloc[idx].y1_seg

        #load initial timepoint
        if self.mode == 'train':
            try:
                output = self.transforms_rand({'img': y0_img, 'label': y0_seg})
                y0_img, y0_seg = output['img'], output['label']
            except:
                output = self.transforms_rand({'img': y1_img, 'label': y1_seg})
                y0_img, y0_seg = output['img'], output['label']

                #y0_seg = '/workspace/radraid/projects/longitudinal_lung/temporal_seg/temporalSeg/templates/template.nii.gz'
                #y0_img =  '/workspace/radraid/projects/longitudinal_lung/temporal_seg/temporalSeg/templates/ct.nii.gz'
                #output = self.transforms({'img': y0_img, 'label': y0_seg})
                #y0_img, y0_seg = output['img'], output['label']
        else:
            output = self.transforms_rand({'img': y1_img, 'label': y1_seg})
            y0_img, y0_seg = output['img'], output['label']
            
        #load final timepoint
        if self.mode == 'train':
            output = self.transforms_rand({'img': y1_img, 'label': y1_seg})
            y1_img, y1_seg = output['img'], output['label']
        else:
            output = self.transforms({'img': y1_img, 'label': y1_seg})
            y1_img, y1_seg = output['img'], output['label']

        #dist_map_tensor = self.disttransform(y1_seg.squeeze().astype(np.int8))
            
        if self.mask:
            set_y1_img = torch.quantile(y1_img, 0.9)
            y1_img = self.random_mask_patches_3d(y1_img, 
                                                 patch_size=(self.mask_size, self.mask_size, self.mask_size), 
                                                 mask_percentage= self.mask_percent, 
                                                 replace = set_y1_img)
            #for _ in range(5):
            #    mask_x, mask_y = np.random.uniform(40,160,2).astype(int)
            #    mask_z = np.random.randint(10,180)
            #    y1_img[:, mask_x:mask_x+45, mask_y:mask_y+45, mask_z:mask_z+45] = set_y1_img

        return y0_img, y0_seg, y1_img, y1_seg#, torch.zeros(1)
    
    @staticmethod
    def random_mask_patches_3d(img, patch_size=(20, 20, 20), mask_percentage=40, replace = 0):

        # Get image dimensions
        _ ,depth, height, width = img.shape

        patches_in_depth = depth // patch_size[0]
        patches_in_height = height // patch_size[1]
        patches_in_width = width // patch_size[2]
        total_patches = patches_in_depth * patches_in_height * patches_in_width

        # Calculate the number of patches to mask
        num_patches_to_mask = int(total_patches * (mask_percentage / 100))

        # Create a binary mask to randomly select patches to mask
        mask = torch.zeros(img.shape, dtype=torch.uint8)

        patch_indices = np.random.choice(range(total_patches), num_patches_to_mask, replace = False)

        for index in patch_indices:
            z_idx = index // (patches_in_height * patches_in_width)
            y_idx = (index % (patches_in_height * patches_in_width)) // patches_in_width
            x_idx = (index % (patches_in_height * patches_in_width)) % patches_in_width

            mask[:, z_idx * patch_size[0]:(z_idx + 1) * patch_size[0],
                y_idx * patch_size[1]:(y_idx + 1) * patch_size[1],
                x_idx * patch_size[2]:(x_idx + 1) * patch_size[2]] = 1

        # Apply the mask to the 3D image
        img[mask] = replace

        return img
