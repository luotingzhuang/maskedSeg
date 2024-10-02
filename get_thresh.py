import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.train_model import Trainer
from utils.train_utils import random_mask_patches_3d
from dataset.dataloader import TaskDataset
from monai.data import MetaTensor
import random
import json
from tqdm import tqdm

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
from monai.utils.type_conversion import convert_to_tensor
from utils.metric_utils import DiceCoefficient, Seg_Metirc3d
from monai.transforms.utils import allow_missing_keys_mode, fill_holes
from skimage import  measure
from utils.eval_utils import get_metrics, find_connected_components, return_nodule, reori, get_dice_coefficient
from utils.eval_utils import load_img, plot_overlay, plot_img
from utils.eval_utils import MyArgs
import sys
import pickle 

path = '/workspace/radraid/projects/seg_lung/masked_seg/maskedSeg/results_update'
cleaned_semantic = pd.read_csv('/workspace/radraid/projects/longitudinal_lung/characterization/cleaned_csv/cleaned_semantic.csv')
exp = sys.argv[1]
exp_dir = os.path.join(path,exp)#all_exp[-3])#-1
with open(os.path.join(exp_dir, "args.json"), "r") as file:
    loaded_args = json.load(file)
    
    
args = MyArgs(**loaded_args)

if isinstance(args.mask_size, int):
    args.mask_size = [args.mask_size]

if isinstance(args.mask_percent, int):
    args.mask_percent = [args.mask_percent]

args.offset = True

val_dataset = TaskDataset(
    csv_file=args.csv_file, 
    mode="val",
    mask=args.mask,
    mask_size=args.mask_size,
    mask_percent=args.mask_percent,
    offset= args.offset
)
data = val_dataset.data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Trainer(args).model
model = model.to(device)
model.load_state_dict(torch.load(os.path.join(exp_dir,
                                              'es_checkpoint.pth.tar'))["model"])

transforms = Compose([
    LoadImaged(keys=['img', 'label']),
    EnsureChannelFirstd(keys=['img', 'label']),
    Orientationd(keys=['img', 'label'], axcodes='RAS'),
    Spacingd(keys=['img', 'label'], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
    ThresholdIntensityd(keys = 'img', threshold =-1024.0, above = True, cval = -1024.0),
    ThresholdIntensityd(keys = 'img', threshold = 276.0, above = False, cval = 276.0,),
    NormalizeIntensityd(keys = 'img', subtrahend  = -370.00039267657144, divisor = 436.5998675471528),
    ResizeWithPadOrCropd(keys=['img', 'label'], spatial_size =(224,224,224), mode = 'constant'),
])

threshs  = [i /20 for i in range(20)]


n= 50
all_pids =  np.sort(val_dataset.data.pid.values.astype(str))
all_thresh_dice = []
save_dir = os.path.join(exp_dir,f'val_n{n}')
os.makedirs(save_dir,exist_ok = True)
for pid in tqdm(all_pids):
    y1_img_path = data[data.pid == pid].y1_img.values[0]
    y1_seg_path = data[data.pid == pid].y1_seg.values[0]
    output = transforms({'img': y1_img_path, 'label': y1_seg_path})
    y1_img, y1_seg = output['img'], output['label']

    
    np.random.seed(10)
    random.seed(10)
    bootstrap = []
    
    with torch.no_grad():
        for _ in tqdm(range(n)):
            with torch.cuda.amp.autocast():
                if isinstance(args.mask_size, int):
                    ms = args.mask_size
                else:
                    ms = random.choice(args.mask_size)

                if isinstance(args.mask_percent, int):
                    mp = args.mask_percent
                else:
                    mp = random.choice(args.mask_percent)

                #y1_img_mask = y1_img.clone()
                y1_img_mask = random_mask_patches_3d(y1_img.clone(), 
                                                     patch_size=(ms,ms,ms), 
                                                     mask_percentage=mp, 
                                                     replace = 1.0,
                                                    offset = False)
                y1_img_mask = y1_img.clone()
                y1_img_mask = y1_img_mask.unsqueeze(0).to(device)


                ph_1 = torch.zeros(y1_img_mask.shape).to(device)
                ph_2 = torch.zeros(y1_img_mask.shape).to(device)
                with torch.no_grad():
                    model_input = (ph_1, ph_2, y1_img_mask)
                    model_out = model(model_input)
                    model_out_sigmoid = torch.sigmoid(model_out[0])
                    bootstrap.append(model_out_sigmoid)
        bootstrap = torch.stack(bootstrap)
        pred = torch.mean(bootstrap,dim =0).squeeze(0).detach().cpu().float()
        gt = y1_seg.squeeze(0).detach().cpu().float()
        
        thresh_dice = []
        for thresh in threshs:
            dice = get_dice_coefficient(pred > thresh, gt)
            thresh_dice.append(dice)
            
        all_thresh_dice.append(thresh_dice)


    np.savez(os.path.join(save_dir,f'{pid}.npz') ,**{'pred': pred, 'gt': gt})


with open(os.path.join(save_dir,f'saved_get_thresh_n{n}.pkl'), 'wb') as f:
    pickle.dump(all_thresh_dice, f)