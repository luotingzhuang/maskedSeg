import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import traceback
import random
import torch
import argparse
from utils.train_model import Trainer
from utils.train_utils import random_mask_patches_3d
from skimage import measure
from monai.transforms.utils import allow_missing_keys_mode
from monai.data import MetaTensor
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    Spacingd,
    ThresholdIntensityd,
    NormalizeIntensityd,
    SaveImage,
)


class MyArgs:
    def __init__(self, **kwargs):
        # Assign all keyword arguments to attributes
        self.__dict__.update(kwargs)
        self.sche = None
        self.max_epoch = 50


def find_connected_components(image, size_threshold):
    """
    Find connected components in the image

    Args:
        image (np.array): input image
        size_threshold (int): size threshold for the connected components

    """
    # Convert the image to binary using a suitable threshold
    binary_image = image > 0.5  # Adjust the threshold as needed

    # Label connected components
    labeled_image = measure.label(binary_image)

    # Measure properties of labeled regions
    regions = measure.regionprops(labeled_image)

    # Filter components based on size threshold
    large_components = [region for region in regions if region.area > size_threshold]

    return labeled_image, large_components


def eval(
    pid: str,
    img_path: str,
    model: torch.nn.Module,
    n: int = 100,
    thresh: float = 0.6,
    result_dir: str = None,
    save_as="numpy",
):
    """
    Generate segmentation on the image and save the results

    Args:
        pid (str): patient id
        img_path (str): path to the image
        model (nn.Module): model to use for the segmentation
        n (int): number of masked samples
        thresh (float): threshold for the segmentation
        result_dir (str): path to save the results
        save_as (str): save as numpy or nifti

    Returns:
        None
    """
    os.makedirs(f"{result_dir}/{pid}", exist_ok=True)
    output = transforms({"img": img_path, "label": img_path})
    img, seg_ori = output["img"], output["label"]
    set_img = 1

    # n different masked images
    bootstrap = []
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

            img_mask = random_mask_patches_3d(
                img.clone(),
                patch_size=(ms, ms, ms),
                mask_percentage=mp,
                replace=set_img,
                offset=False,
            )
            img_mask = img_mask.unsqueeze(0).to(device)
            with torch.no_grad():
                model_out = model(img_mask)
                model_out_sigmoid = torch.sigmoid(model_out[0])
                bootstrap.append(model_out_sigmoid)

    bootstrap = torch.stack(bootstrap)
    mean_sigmoid = torch.mean(bootstrap, dim=0)
    std_sigmoid = torch.std(bootstrap, dim=0)

    # invert the transformation so that the segmentation is in original size
    output = transforms({"img": img_path, "label": img_path})
    with allow_missing_keys_mode(transforms):
        seg = (mean_sigmoid > thresh).squeeze(0).detach().cpu().float()
        seg = MetaTensor(seg).copy_meta_from(seg_ori)
        seg.applied_operations = output["label"].applied_operations
        inverted_seg = transforms.inverse({"label": seg})
    mean_seg = inverted_seg["label"].squeeze()
    mean_seg_copy, large_components = find_connected_components(
        mean_seg.clone(), 1000000
    )
    mean_seg_copy[mean_seg_copy > 0] = 1
    mean_seg = MetaTensor(mean_seg_copy).copy_meta_from(mean_seg)

    output = transforms({"img": img_path, "label": img_path})
    with allow_missing_keys_mode(transforms):
        seg = std_sigmoid.squeeze(0).detach().cpu().float()
        seg = MetaTensor(seg).copy_meta_from(seg_ori)
        seg.applied_operations = output["label"].applied_operations
        inverted_seg = transforms.inverse({"label": seg})
    std_seg = inverted_seg["label"].squeeze()

    # save the results
    if save_as == "numpy":
        np.savez(
            os.path.join(result_dir, f"{pid}.npz"),
            **{
                "mean_seg": mean_seg.numpy(),
                "std_seg": std_seg.numpy(),
            },
        )
    elif save_as == "nifti":
        saver = SaveImage()
        saver(mean_seg, filename=f"{result_dir}/{pid}/mean")
        saver(std_seg, filename=f"{result_dir}/{pid}/std")


def init_model(args, device, exp_dir):
    model = Trainer(args).model
    model = model.to(device)
    model.load_state_dict(
        torch.load(os.path.join(exp_dir, "es_checkpoint.pth.tar"))["model"]
    )
    model.eval()
    return model


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


# argparse
def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file",
        type=str,
        default="./dataset_csv/sample.csv",
        help="path to the csv file",
    )
    parser.add_argument(
        "--result_dir", type=str, default="./output", help="path to save the results"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="./model_weights",
        help="path to the experiment directory",
    )
    parser.add_argument("--n", type=int, default=100, help="number of masked samples")
    parser.add_argument(
        "--thresh", type=float, default=0.55, help="threshold for the segmentation"
    )
    parser.add_argument(
        "--save_as",
        type=str,
        default="nifti",
        choices=["nifti", "numpy"],
        help="save as nifti or numpy",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("Starting...")
    eval_args = load_args()

    print("Model arguments loaded...")
    with open(os.path.join(eval_args.exp_dir, "args.json"), "r") as file:
        loaded_args = json.load(file)
    args = MyArgs(**loaded_args)
    args.totalseg_weight = "./model_weights/checkpoint_final.pth"
    if not os.path.exists(args.totalseg_weight):
        raise FileNotFoundError(
            f"Totalsegmentator weight not found: {args.totalseg_weight}"
        )

    print("Initialize model...")
    model = init_model(args, device, eval_args.exp_dir)

    csv = pd.read_csv(eval_args.csv_file)
    for _, row in csv.iterrows():
        print(f"Processing {row['pid']}")
        pid = row.pid
        img_path = row.image_path
        try:
            eval(
                pid,
                img_path,
                model,
                eval_args.n,
                eval_args.thresh,
                eval_args.result_dir,
                eval_args.save_as,
            )
        except:
            print(f"Error in {pid}")
            print(traceback.format_exc())

    print("Done...")
