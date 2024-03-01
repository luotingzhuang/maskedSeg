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
    fig, ax = plt.subplots(1, len(arrs), figsize=(len(arrs) * 5, 10))
    for i, (title, arr) in enumerate(arrs.items()):
        ax[i].imshow(arr, cmap="gray")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(title)

    plt.show()


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sensitivity_path(lung, path):
    return torch.sum(torch.mul(lung, path)) / torch.sum(path)


def precision_path(lung, path):
    return torch.sum(torch.mul(lung, path)) / torch.sum(lung)


def hd95(pred, gt):
    return compute_hausdorff_distance(
        y_pred=pred, y=gt, percentile=95, spacing =1.5
    )


def eval(all_cases, idx, model, n=100, thresholds=[0.5]):
    y1_img_path = data[data.pid == int(all_cases[idx])].y1_img.values[0]
    y1_seg_path = os.path.join(seg_mod_path, f"{all_cases[idx]}.nii.gz")

    img_path = "/workspace/radraid/projects/longitudinal_lung/NLST_CT_longitudinal"

    case_path = os.path.join(img_path, str(all_cases[idx]))
    years = os.listdir(case_path)
    year = max(years)
    lungmask_path = os.path.join(case_path, year, "nodule_0", "lungmask", "lung.nii.gz")
    totalseg_path = os.path.join(
        case_path, year, "nodule_0", "totalsegmentator", "lung.nii.gz"
    )

    output = transforms({"img": y1_img_path, "label": y1_seg_path})
    y1_img, y1_seg = output["img"], output["label"]

    output = transforms({"img": y1_img_path, "label": lungmask_path})
    _, y1_seg_lungmask = output["img"], output["label"]

    output = transforms({"img": y1_img_path, "label": totalseg_path})
    _, y1_seg_total = output["img"], output["label"]

    set_y1_img = torch.quantile(y1_img, 0.9)

    # bootstrap
    np.random.seed(10)
    bootstrap = []
    for _ in tqdm(range(n)):
        y1_img_mask = random_mask_patches_3d(
            y1_img.clone(),
            patch_size=(args.mask_size, args.mask_size, args.mask_size),
            mask_percentage=args.mask_percent,
            replace=set_y1_img,
        )
        y1_img_mask = y1_img_mask.unsqueeze(0).to(device)
        ph_1 = torch.zeros(y1_img_mask.shape).to(device)
        ph_2 = torch.zeros(y1_img_mask.shape).to(device)
        with torch.no_grad():
            model_input = (ph_1, ph_2, y1_img_mask)
            model_out = model(model_input)
            model_out_sigmoid = torch.sigmoid(model_out[0])
            bootstrap.append(model_out_sigmoid)

    (
        dice_lung_thresh,
        fnr_lung_thresh,
        fpr_lung_thresh,
        assd_lung_thresh,
        msd_lung_thresh,
        hd95_lung_thresh,
    ) = ([], [], [], [], [], [])

    # use different thresholds
    for thresh in thresholds:
        pred = torch.stack(bootstrap) > thresh
        count_ones = torch.sum(pred, dim=0)
        majority_pred = count_ones >= n / 2
        mean_sigmoid = torch.mean(torch.stack(bootstrap), dim=0)

        mean_metric = Seg_Metirc3d(
            (y1_seg > 0).squeeze().cpu().numpy(),
            (mean_sigmoid > thresh).squeeze().cpu().numpy(),
            (1.5, 1.5, 1.5),
        )

        majority_metric = Seg_Metirc3d(
            (y1_seg > 0).squeeze().cpu().numpy(),
            majority_pred.squeeze().cpu().numpy(),
            (1.5, 1.5, 1.5),
        )

        lungmask_metric = Seg_Metirc3d(
            (y1_seg > 0).squeeze().cpu().numpy(),
            (y1_seg_lungmask > 0).squeeze().cpu().numpy(),
            (1.5, 1.5, 1.5),
        )

        totalseg_metric = Seg_Metirc3d(
            (y1_seg > 0).squeeze().cpu().numpy(),
            (y1_seg_total > 0).squeeze().cpu().numpy(),
            (1.5, 1.5, 1.5),
        )

        dice_lung = pd.DataFrame(
            {
                "metric": "dice_lung",
                "thresh": thresh,
                "model_mean": [mean_metric.get_dice_coefficient()[0]],
                "model_majority": [majority_metric.get_dice_coefficient()[0]],
                "lungmask": [lungmask_metric.get_dice_coefficient()[0]],
                "totalseg": [totalseg_metric.get_dice_coefficient()[0]],
            }
        )

        fnr_lung = pd.DataFrame(
            {
                "metric": "fnr_lung",
                "thresh": thresh,
                "model_mean": [mean_metric.get_FNR()],
                "model_majority": [majority_metric.get_FNR()],
                "lungmask": [lungmask_metric.get_FNR()],
                "totalseg": [totalseg_metric.get_FNR()],
            }
        )

        fpr_lung = pd.DataFrame(
            {
                "metric": "fpr_lung",
                "thresh": thresh,
                "model_mean": [mean_metric.get_FPR()],
                "model_majority": [majority_metric.get_FPR()],
                "lungmask": [lungmask_metric.get_FPR()],
                "totalseg": [totalseg_metric.get_FPR()],
            }
        )

        assd_lung = pd.DataFrame(
            {
                "metric": "assd_lung",
                "thresh": thresh,
                "model_mean": [mean_metric.get_ASSD()],
                "model_majority": [majority_metric.get_ASSD()],
                "lungmask": [lungmask_metric.get_ASSD()],
                "totalseg": [totalseg_metric.get_ASSD()],
            }
        )

        msd_lung = pd.DataFrame(
            {
                "metric": "msd_lung",
                "thresh": thresh,
                "model_mean": [mean_metric.get_MSD()],
                "model_majority": [majority_metric.get_MSD()],
                "lungmask": [lungmask_metric.get_MSD()],
                "totalseg": [totalseg_metric.get_MSD()],
            }
        )

        hd95 = HausdorffDistanceMetric(percentile=95, reduction="mean")
        #import pdb; pdb.set_trace()
        hd95_lung = pd.DataFrame(
            {
                "metric": "hd95_lung",
                "thresh": thresh,
                "model_mean": [
                    hd95(
                        (mean_sigmoid > thresh).permute(4,0,1,2,3),
                        (y1_seg > 0).unsqueeze(0).permute(4,0,1,2,3).to(device),
                    ).nanmean().detach().cpu().numpy().item()
                ],
                "model_majority": [
                    hd95(
                        majority_pred.permute(4,0,1,2,3),
                        (y1_seg > 0).unsqueeze(0).permute(4,0,1,2,3).to(device),
                    ).nanmean().detach().cpu().numpy().item()
                ],
                "lungmask": [
                    hd95(
                        y1_seg_lungmask.unsqueeze(0).permute(4,0,1,2,3).to(device),
                        (y1_seg > 0).unsqueeze(0).permute(4,0,1,2,3).to(device),
                    ).nanmean().detach().cpu().numpy().item()
                ],
                "totalseg": [
                    hd95(
                        y1_seg_total.unsqueeze(0).permute(4,0,1,2,3).to(device),
                        (y1_seg > 0).unsqueeze(0).permute(4,0,1,2,3).to(device),
                    ).nanmean().detach().cpu().numpy().item()
                ],
            }
        )

        dice_lung_thresh.append(dice_lung)
        fnr_lung_thresh.append(fnr_lung)
        fpr_lung_thresh.append(fpr_lung)
        assd_lung_thresh.append(assd_lung)
        msd_lung_thresh.append(msd_lung)
        hd95_lung_thresh.append(hd95_lung)

    dice_lung_thresh = pd.concat(dice_lung_thresh)
    fnr_lung_thresh = pd.concat(fnr_lung_thresh)
    fpr_lung_thresh = pd.concat(fpr_lung_thresh)
    assd_lung_thresh = pd.concat(assd_lung_thresh)
    msd_lung_thresh = pd.concat(msd_lung_thresh)
    hd95_lung_thresh = pd.concat(hd95_lung_thresh)

    return pd.concat(
        [
            dice_lung_thresh,
            fnr_lung_thresh,
            fpr_lung_thresh,
            assd_lung_thresh,
            msd_lung_thresh,
            hd95_lung_thresh,
        ]
    ).sort_values(["metric", "thresh"])


if __name__ == "__main__":

    path = "/workspace/radraid/projects/longitudinal_lung/temporal_seg/temporalSeg/results_random_mask_new"

    cleaned_semantic = pd.read_csv(
        "/workspace/radraid/projects/longitudinal_lung/characterization/cleaned_csv/cleaned_semantic.csv"
    )

    all_exp = [
        #"exp_adam_lr0.001_bs1_us64_seed0_DiceCELoss_mask7_70_freeze",
        #"exp_adam_lr0.001_bs1_us64_seed0_DiceCELoss_mask14_70_freeze",
        "exp_adam_lr0.001_bs1_us64_seed0_DiceCELoss_mask28_70_freeze",
        #'exp_adam_lr0.001_bs1_us64_seed0_DiceCELoss_mask2_70_freeze',
        "exp_adam_lr0.001_bs1_us64_seed0_DiceCELoss_mask4_70_freeze",
    ]
    for exp in tqdm(all_exp):
        exp_dir = os.path.join(path, exp)
        with open(os.path.join(exp_dir, "args.json"), "r") as file:
            loaded_args = json.load(file)

        args = MyArgs(**loaded_args)

        test_dataset = TaskDataset(csv_file=args.csv_file, mode="test")
        data = test_dataset.data

        seg_mod_path = "/workspace/radraid/projects/longitudinal_lung/seg_mod"
        all_cases = [i.split(".")[0] for i in os.listdir(seg_mod_path) if "nii.gz" in i]
        all_cases.sort()
        # all_cases = all_cases[0:9]

        # model
        model = Trainer(args).model
        model = model.to(device)
        model.load_state_dict(
            torch.load(os.path.join(exp_dir, "es_checkpoint.pth.tar"))["model"]
        )
        model.eval()
        thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        result_dir = os.path.join(
            "/workspace/radraid/projects/longitudinal_lung/temporal_seg/temporalSeg",
            "model_results",
            args.exp_dir.split("/")[-1],
        )
        os.makedirs(result_dir, exist_ok=True)

        for idx in tqdm(range(len(all_cases))):
            csv_file = eval(all_cases, idx, model, n=100, thresholds=thresh)
            csv_file["case"] = all_cases[idx]
            csv_file.to_csv(os.path.join(result_dir, all_cases[idx] + ".csv"))
