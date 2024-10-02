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
from utils.metric_utils import Seg_Metirc3d
from monai.metrics import HausdorffDistanceMetric
from monai.transforms.utils import allow_missing_keys_mode, fill_holes
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
    ScaleIntensityd,
)
import random
from utils.eval_utils import get_metrics, find_connected_components, return_nodule
from utils.eval_utils import MyArgs
import sys

transforms = Compose(
    [
        LoadImaged(keys=["img", "label"]),
        EnsureChannelFirstd(keys=["img", "label"]),
        Orientationd(keys=["img", "label"], axcodes="RAS"),
        Spacingd(
            keys=["img", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")
        ),
        ThresholdIntensityd(keys="img", threshold=-1024.0, above=True, cval=-1024.0),
        ThresholdIntensityd(
            keys="img",
            threshold=276.0,
            above=False,
            cval=276.0,
        ),
        NormalizeIntensityd(
            keys="img", subtrahend=-370.00039267657144, divisor=436.5998675471528
        ),
        ResizeWithPadOrCropd(
            keys=["img", "label"], spatial_size=(224, 224, 224), mode="constant"
        ),
    ]
)

simple_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        Orientationd(keys=["img"], axcodes="RAS"),
    ]
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval(all_cases, idx, model, n=100, thresholds=[0.6], result_dir=None, pred_type = 'bootstrap'):
    print(all_cases[idx])
    y1_img_path = data[data.pid == str(all_cases[idx])].y1_img.values[0]
    y1_seg_path = os.path.join(seg_mod_path, f"{all_cases[idx]}.nii.gz")
    output = transforms({"img": y1_img_path, "label": y1_seg_path})
    y1_img, y1_seg = output["img"], output["label"]
    set_y1_img = 1  # torch.quantile(y1_img, 0.9)


    cleaned_semantic_pid = cleaned_semantic[cleaned_semantic.pid == int(all_cases[idx])][['pid','nodule_id','lobar_location','axial_location','image_number','longest_axial_diameter_(mm)']]


    # load the image
    output = simple_transforms({"img": y1_img_path})
    original_img = output["img"].squeeze().numpy()
    original_img[original_img < -1000] = -1000
    original_img[original_img > 1000] = 1000
    original_img = np.flip(original_img, 0)
    img_path = "/workspace/radraid/projects/longitudinal_lung/NLST_CT_longitudinal"
    # pid = 111832

    case_path = os.path.join(img_path, str(all_cases[idx]))
    years = os.listdir(case_path)
    year = max(years)
    case_path_year_lungmask = os.path.join(
        case_path, year, "nodule_0", "lungmask", "lung.nii.gz"
    )
    case_path_year_total = os.path.join(
        case_path, year, "nodule_0", "totalsegmentator", "lung.nii.gz"
    )

    output = simple_transforms({"img": case_path_year_lungmask})
    lungmask_seg = output["img"].squeeze().numpy()
    lungmask_seg = np.flip(lungmask_seg, 0)

    output = simple_transforms({"img": case_path_year_total})
    total_seg = output["img"].squeeze().numpy()
    total_seg = np.flip(total_seg, 0)

    output = simple_transforms({"img": y1_seg_path})
    corrected_seg = output["img"].squeeze().numpy()
    corrected_seg = np.flip(corrected_seg, 0)

    # identify the nodule location
    nodule_loc_pid = nodule_loc[nodule_loc.pid == int(all_cases[idx])]

    x = nodule_loc_pid.x.item()
    y = nodule_loc_pid.y.item()
    z = nodule_loc_pid.z.item()

    # bootstrap
    np.random.seed(10)
    
    # obtain the longest axial diameter of the nodule
    if cleaned_semantic_pid[cleaned_semantic_pid.image_number == z].empty:
        selected_n = cleaned_semantic_pid[
            cleaned_semantic_pid.image_number == original_img.shape[2] - z
        ]
        r = int(selected_n["longest_axial_diameter_(mm)"].item() // 2)
    else:
        selected_n = cleaned_semantic_pid[cleaned_semantic_pid.image_number == z]
        r = int(selected_n["longest_axial_diameter_(mm)"].item() // 2)
    voxel_spacing = output["img"].meta["pixdim"][1:4]
    #r_x, r_y, r_z = (r / voxel_spacing).astype(
    #    int
    #)  # convert the diameter to voxel space
    r_x, r_y, r_z = np.ceil(r / voxel_spacing).astype(int)
    
    
    only_gt = False
    bootstrap = []
    if pred_type == 'bootstrap':
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

                # y1_img_mask = y1_img.clone()
                y1_img_mask = random_mask_patches_3d(
                    y1_img.clone(),
                    patch_size=(ms, ms, ms),
                    mask_percentage=mp,
                    replace=set_y1_img,
                    offset=False,
                )
                y1_img_mask = y1_img_mask.unsqueeze(0).to(device)
                ph_1 = torch.zeros(y1_img_mask.shape).to(device)
                ph_2 = torch.zeros(y1_img_mask.shape).to(device)
                with torch.no_grad():
                    model_input = (ph_1, ph_2, y1_img_mask)
                    model_out = model(model_input)
                    model_out_sigmoid = torch.sigmoid(model_out[0])
                    bootstrap.append(model_out_sigmoid)

        bootstrap = torch.stack(bootstrap)

        for thresh in thresholds:
            mean_sigmoid = torch.mean(bootstrap, dim=0)
            std_sigmoid = torch.std(bootstrap, dim=0)

            # invert the transformation so that the segmentation is in original size
            output = transforms({"img": y1_img_path, "label": y1_seg_path})
            with allow_missing_keys_mode(transforms):
                seg = (mean_sigmoid > thresh).squeeze(0).detach().cpu().float()
                seg = MetaTensor(seg).copy_meta_from(y1_seg)
                seg.applied_operations = output["label"].applied_operations
                inverted_seg = transforms.inverse({"label": seg})
            mean_seg = inverted_seg["label"].squeeze().numpy()
            mean_seg, large_components = find_connected_components(mean_seg, 1000000)
            mean_seg[mean_seg > 0] = 1

            output = transforms({"img": y1_img_path, "label": y1_seg_path})
            with allow_missing_keys_mode(transforms):
                seg = std_sigmoid.squeeze(0).detach().cpu().float()
                seg = MetaTensor(seg).copy_meta_from(y1_seg)
                seg.applied_operations = output["label"].applied_operations
                inverted_seg = transforms.inverse({"label": seg})
            std_seg = inverted_seg["label"].squeeze().numpy()

            # get the nodule 3d patch region
            mean_seg_nodule = return_nodule(mean_seg, x, y, z, r_x, r_y, r_z)

            np.savez(
                os.path.join(result_dir, f"mean_thresh{thresh}", f"{all_cases[idx]}.npz"),
                **{
                    "mean_seg": mean_seg,
                    "std_seg": std_seg,
                    "mean_seg_nodule": mean_seg_nodule,
                },
            )
    elif pred_type == 'original':
        y1_img = y1_img.unsqueeze(0).to(device)
        ph_1 = torch.zeros(y1_img.shape).to(device)
        ph_2 = torch.zeros(y1_img.shape).to(device)
        with torch.no_grad():
            model_input = (ph_1, ph_2, y1_img)
            model_out = model(model_input)
            model_out_sigmoid = torch.sigmoid(model_out[0])
            
            
        for thresh in thresholds:
            # invert the transformation so that the segmentation is in original size
            output = transforms({"img": y1_img_path, "label": y1_seg_path})
            with allow_missing_keys_mode(transforms):
                seg = (model_out_sigmoid > thresh).squeeze(0).detach().cpu().float()
                seg = MetaTensor(seg).copy_meta_from(y1_seg)
                seg.applied_operations = output["label"].applied_operations
                inverted_seg = transforms.inverse({"label": seg})
            mean_seg = inverted_seg["label"].squeeze().numpy()
            mean_seg, large_components = find_connected_components(mean_seg, 1000000)
            mean_seg[mean_seg > 0] = 1

            # get the nodule 3d patch region
            mean_seg_nodule = return_nodule(mean_seg, x, y, z, r_x, r_y, r_z)

            np.savez(
                os.path.join(result_dir, f"original_thresh{thresh}", f"{all_cases[idx]}.npz"),
                **{
                    "original_seg": mean_seg,
                    "original_seg_nodule": mean_seg_nodule,
                },
            )
            
    else:
        original_img_nodule = return_nodule(original_img, x, y, z, r_x, r_y, r_z)
        lungmask_seg_nodule = return_nodule(lungmask_seg, x, y, z, r_x, r_y, r_z)
        corrected_seg_nodule = return_nodule(corrected_seg, x, y, z, r_x, r_y, r_z)
        total_seg_nodule = return_nodule(total_seg, x, y, z, r_x, r_y, r_z)

        os.makedirs(os.path.join(result_dir, f"save_img"),exist_ok = True)
        np.savez(
            os.path.join(result_dir, f"save_img", f"{all_cases[idx]}.npz"),
            **{
                "original_img": original_img,
                "corrected_seg": corrected_seg,
                "lungmask_seg": lungmask_seg,
                "total_seg": total_seg,
                "original_img_nodule": original_img_nodule,
                "corrected_seg_nodule":corrected_seg_nodule,
                "lungmask_seg_nodule": lungmask_seg_nodule,
                "total_seg_nodule": total_seg_nodule,
            },
        )


if __name__ == "__main__":

    path = "/workspace/radraid/projects/seg_lung/masked_seg/maskedSeg/results_update"
    nodule_loc = pd.read_csv('/workspace/radraid/projects/seg_lung/nlst_segmentation/seg_mod/nodule_loc.csv')
    cleaned_semantic = pd.read_csv(
        "/workspace/radraid/projects/longitudinal_lung/characterization/cleaned_csv/cleaned_semantic.csv"
    )
    #input the name of experiment
    exp = sys.argv[1]
    pred_type = sys.argv[2]
    print(exp,pred_type)
    exp_dir = os.path.join(path, exp)
    with open(os.path.join(exp_dir, "args.json"), "r") as file:
        loaded_args = json.load(file)

    args = MyArgs(**loaded_args)

    test_dataset = TaskDataset(csv_file=args.csv_file, mode="test")
    data = test_dataset.data

    seg_mod_path = "/workspace/radraid/projects/seg_lung/nlst_segmentation/corrected_segmentations"
    all_cases = nodule_loc.pid.values
    
    only_gt = False
    # model
    model = Trainer(args).model
    model = model.to(device)
    model.load_state_dict(
        torch.load(os.path.join(exp_dir, "es_checkpoint.pth.tar"))["model"]
    )
    model.eval()
    threshs = [0.5 , 0.55, 0.6, 0.65, 0.7]
    n = 100
    result_dir = os.path.join(
        "/workspace/radraid/projects/seg_lung/masked_seg/maskedSeg",
        "model_results",
        args.exp_dir.split("/")[-1],
        f"pred_seg_{n}",
    )
    os.makedirs(result_dir, exist_ok=True)
    for thresh in threshs:
        if pred_type == 'bootstrap':
            os.makedirs(os.path.join(result_dir, f"mean_thresh{thresh}"), exist_ok=True)
        elif pred_type == 'original':
            os.makedirs(os.path.join(result_dir, f"original_thresh{thresh}"), exist_ok=True)

    for idx in tqdm(range(len(all_cases))):
        eval(
            all_cases, idx, model, n=n, thresholds=threshs, result_dir=result_dir, pred_type = pred_type
        )
