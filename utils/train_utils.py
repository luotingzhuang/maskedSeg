import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import os


def initialize_weights(module):
    # should be modified to initialize weights for different types of layers
    for m in module.modules():
        if isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EarlyStopping:
    def __init__(
        self, warmup: int = 10, patience: int = 20, verbose: bool = True
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.lowest_val_loss = np.Inf
        self.early_stop = False
        self.warmup = warmup

    def __call__(
        self,
        epoch: int,
        val_loss: float,
        model: torch.nn.Module,
        ckpt_path: str = "checkpoint.pt",
    ) -> None:
        if epoch < self.warmup:
            pass
        elif np.isinf(self.lowest_val_loss):
            self.save_checkpoint(val_loss, model, ckpt_path)
            self.lowest_val_loss = val_loss
        elif val_loss > self.lowest_val_loss:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model, ckpt_path)
            self.lowest_val_loss = val_loss
            self.counter = 0

    def save_checkpoint(
        self, val_loss: float, model: torch.nn.Module, ckpt_path: str
    ) -> None:
        if self.verbose:
            print(
                f"Validation loss decreased from {self.lowest_val_loss:.6f} to {val_loss:.6f}. Model saved."
            )
        torch.save({"model": model.state_dict()}, ckpt_path)

def random_mask_patches_3d(img, patch_size=(20, 20, 20), mask_percentage=40, replace = 0, offset = False):

    # Get image dimensions
    if offset:
        offset_x, offset_y, offset_z  = np.random.randint(0, patch_size[0], size =3)
    else:
        offset_x, offset_y, offset_z  = 0, 0 , 0
        
    depth = img.shape[1] - offset_x
    height = img.shape[2] - offset_y
    width = img.shape[3] - offset_z
    
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
        mask[:, z_idx * patch_size[0] + offset_x:(z_idx + 1) * patch_size[0] + offset_x,
            y_idx * patch_size[1] + offset_y:(y_idx + 1) * patch_size[1] + offset_y,
            x_idx * patch_size[2] + offset_z:(x_idx + 1) * patch_size[2] + offset_z] = 1

    # Apply the mask to the 3D image
    img[mask.bool()] = replace

    return img
