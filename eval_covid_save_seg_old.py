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
import json
from tqdm import tqdm
from utils.metric_utils import DiceCoefficient
from monai.transforms.utils import allow_missing_keys_mode
from monai.data import MetaTensor
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
    RandAffined,
    ScaleIntensityd
)
import numpy as np

class MyArgs:
    def __init__(self, **kwargs):
        # Assign all keyword arguments to attributes
        self.__dict__.update(kwargs)
        self.sche = None
        self.max_epoch = 50

def load_img(path):
    sitk_img = sitk.ReadImage(path)
    sit_arr = sitk.GetArrayFromImage(sitk_img)
    return sitk_img, sit_arr

# plot y1_img, y1_seg, pred_seg

def plot_img(arrs):
    fig, ax = plt.subplots(1,len(arrs) , figsize = (len(arrs) * 5, 10))
    for i, (title, arr) in enumerate(arrs.items()):
        ax[i].imshow(arr,cmap = 'gray')
        ax[i].set_xticks([]) 
        ax[i].set_yticks([]) 
        ax[i].set_title(title)
        
    plt.show()



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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def sensitivity_path(lung,path):
    return torch.sum(torch.mul(lung,path)) / torch.sum(path)

def precision_path(lung,path):
    return torch.sum(torch.mul(lung,path)) / torch.sum(lung)



def eval(all_cases, idx, model, n = 100, thresholds = [0.5]):

    y1_img_path = os.path.join(covid_path, 'COVID-19-CT-Seg_20cases', all_cases[idx])
    y1_seg_path = os.path.join(covid_path, 'Lung_and_Infection_Mask', all_cases[idx])

    output = transforms({'img': y1_img_path, 'label': y1_seg_path})
    y1_img, y1_seg = output['img'], output['label']


    set_y1_img = torch.quantile(y1_img, 0.9)

    #bootstrap
    np.random.seed(10)
    bootstrap = []
    for _ in tqdm(range(n)):
        with torch.cuda.amp.autocast():
            y1_img_mask = random_mask_patches_3d(y1_img.clone(), 
                                                patch_size=(args.mask_size, args.mask_size, args.mask_size), 
                                                mask_percentage=args.mask_percent, replace = set_y1_img)
            y1_img_mask = y1_img_mask.unsqueeze(0).to(device)
            ph_1 = torch.zeros(y1_img_mask.shape).to(device)
            ph_2 = torch.zeros(y1_img_mask.shape).to(device)
            with torch.no_grad():
                model_input = (ph_1, ph_2, y1_img_mask)
                model_out = model(model_input)
                model_out_sigmoid = torch.sigmoid(model_out[0])
                bootstrap.append(model_out_sigmoid)

    pred = torch.stack(bootstrap) > thresh
    count_ones = torch.sum(pred, dim=0)
    majority_pred = count_ones >= n / 2
    mean_sigmoid = torch.mean(torch.stack(bootstrap), dim=0)

    output = transforms({'img': y1_img_path, 'label': y1_seg_path})

    with allow_missing_keys_mode(transforms):
        majority_seg = majority_pred.squeeze(0).detach().cpu().float()
        majority_seg = MetaTensor(majority_seg).copy_meta_from(y1_seg)
        majority_seg.applied_operations =output['label'].applied_operations
        inverted_seg = transforms.inverse({'label':majority_seg})
        
    majority_seg = inverted_seg['label'].squeeze().numpy()

    output = transforms({'img': y1_img_path, 'label': y1_seg_path})

    with allow_missing_keys_mode(transforms):
        seg = (mean_sigmoid > thresh).squeeze(0).detach().cpu().float()
        seg = MetaTensor(seg).copy_meta_from(y1_seg)
        seg.applied_operations =output['label'].applied_operations
        inverted_seg = transforms.inverse({'label':seg})
        
    mean_seg = inverted_seg['label'].squeeze().numpy()

    return majority_seg, mean_seg

if __name__ == "__main__":

    path = '/workspace/radraid/projects/longitudinal_lung/temporal_seg/temporalSeg/results_random_mask_new'
    #all_exp = [i for i in os.listdir(path) if 'ipynb' not in i]
    #all_exp.sort()

    all_exp = [
        'exp_adam_lr0.001_bs1_us64_seed0_DiceCELoss_mask14_70_freeze',
        'exp_adam_lr0.001_bs1_us64_seed0_DiceCELoss_mask28_70_freeze',
        #'exp_adam_lr0.001_bs1_us64_seed0_DiceCELoss_mask2_70_freeze',
        'exp_adam_lr0.001_bs1_us64_seed0_DiceCELoss_mask4_70_freeze',
        'exp_adam_lr0.001_bs1_us64_seed0_DiceCELoss_mask7_70_freeze']
    for exp in tqdm(all_exp):
        exp_dir = os.path.join(path,exp)
        with open(os.path.join(exp_dir, "args.json"), "r") as file:
            loaded_args = json.load(file)
            
        args = MyArgs(**loaded_args)
        covid_path = '/workspace/radraid/projects/longitudinal_lung/covid19_20'
        all_cases = [i for i in os.listdir(os.path.join(covid_path, 'COVID-19-CT-Seg_20cases' )) if 'nii.gz' in i]
        all_cases.sort()
        #all_cases = all_cases[0:9]

        #model
        model = Trainer(args).model
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(exp_dir,
                                                    'es_checkpoint.pth.tar'))["model"])
        model.eval()
        thresh = 0.6

        result_dir = os.path.join(covid_path, 'model_results', args.exp_dir.split('/')[-1])
        os.makedirs(result_dir, exist_ok = True)
        os.makedirs(os.path.join(result_dir, f"majority_thresh{thresh}"), exist_ok = True)
        os.makedirs(os.path.join(result_dir, f"mean_thresh{thresh}"), exist_ok = True)


        for idx in tqdm(range(len(all_cases))):
            majority_seg, mean_seg = eval(all_cases, idx, model, n=100, thresholds=thresh)

            np.save(os.path.join(result_dir,f'majority_thresh{thresh}', f"{all_cases[idx].split('.')[0]}.npy"), majority_seg)
            np.save(os.path.join(result_dir,f'mean_thresh{thresh}' ,f"{all_cases[idx].split('.')[0]}.npy"), mean_seg)