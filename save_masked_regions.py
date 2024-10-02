import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from argparse import ArgumentParser
import gzip
import pickle
import multiprocessing

def random_mask_sphere_3d(i, img_size = (1, 224,224,224), diameter=[28], mask_percentage=70,offset = False, random_diameter = False):
    
    
    diameter_max = int(np.mean(diameter))
    radius_max = diameter_max//2
    
    print(i)
    # Get image dimensions
    if offset:
        offset_x, offset_y, offset_z  = np.random.randint(-diameter_max//2, diameter_max//2, size =3)
    else:
        offset_x, offset_y, offset_z  = 0, 0 , 0

    depth = img_size[1] #- offset_x
    height = img_size[2] #- offset_y
    width = img_size[3] #- offset_z

    patches_in_depth = depth // diameter_max
    patches_in_height = height // diameter_max
    patches_in_width = width // diameter_max
    total_patches = patches_in_depth * patches_in_height * patches_in_width

    # Calculate the number of patches to mask
    num_patches_to_mask = int(total_patches * (mask_percentage / 100))

    # Create a binary mask to randomly select patches to mask
    mask = torch.zeros(img_size, dtype=torch.uint8).to('cuda')
    patch_indices = np.random.choice(range(total_patches), num_patches_to_mask, replace = False)
    

    for index in patch_indices:
        if random_diameter:
            diameter_chosen = np.random.choice(diameter, 1)[0]
        else:
            diameter_chosen = diameter_max
            
        radius_chosen = diameter_chosen//2
            
        z_idx = index // (patches_in_height * patches_in_width) * diameter_max + radius_max - offset_x
        y_idx = (index % (patches_in_height * patches_in_width)) // patches_in_width * diameter_max + radius_max - offset_y
        x_idx = (index % (patches_in_height * patches_in_width)) % patches_in_width * diameter_max + radius_max - offset_z
        z = torch.arange(img_size[1])[:, None, None].to('cuda')
        y = torch.arange(img_size[2])[None, :, None].to('cuda')
        x = torch.arange(img_size[3])[None, None, :].to('cuda')
        
        distances = torch.sqrt((z - z_idx ) ** 2 + (y - y_idx) ** 2 + (x - x_idx ) ** 2)
        mask[0][distances <= radius_chosen] = 1
        
    with gzip.open(os.path.join(save_dir,f'mask_{i}.pkl'), 'wb') as f:
        pickle.dump(mask.detach().cpu().numpy(), f)


def random_mask_patches_3d(i, img_size, length, mask_percentage=40, offset = False, random_diameter = False):
    print(i)
    length_max = int(np.mean(length))
    # Get image dimensions
    if offset:
        offset_x, offset_y, offset_z  = np.random.randint(-length_max//2, length_max//2, size =3)
    else:
        offset_x, offset_y, offset_z  = 0, 0 , 0
        
    depth = img_size[1] #- offset_x
    height = img_size[2] #- offset_y
    width = img_size[3] #- offset_z
    
    patches_in_depth = depth // length_max
    patches_in_height = height // length_max
    patches_in_width = width // length_max
    total_patches = patches_in_depth * patches_in_height * patches_in_width

    # Calculate the number of patches to mask
    num_patches_to_mask = int(total_patches * (mask_percentage / 100))

    # Create a binary mask to randomly select patches to mask
    mask = torch.zeros(img_size, dtype=torch.uint8).to('cuda')

    patch_indices = np.random.choice(range(total_patches), num_patches_to_mask, replace = False)

    for index in patch_indices:
        if random_diameter:
            length_chosen = np.random.choice(length, 1)[0]
        else:
            length_chosen = length_max
        z_idx = index // (patches_in_height * patches_in_width)
        y_idx = (index % (patches_in_height * patches_in_width)) // patches_in_width
        x_idx = (index % (patches_in_height * patches_in_width)) % patches_in_width
        mask[:, z_idx * length_chosen + offset_x:(z_idx + 1) * length_chosen + offset_x,
            y_idx * length_chosen + offset_y:(y_idx + 1) * length_chosen + offset_y,
            x_idx * length_chosen + offset_z:(x_idx + 1) * length_chosen + offset_z] = 1

    with gzip.open(os.path.join(save_dir,f'mask_{i}.pkl'), 'wb') as f:
        pickle.dump(mask.detach().cpu().numpy(), f)


parser = ArgumentParser()
parser.add_argument("--shape",type=str, default='sphere', choices=['sphere', 'cube'])
parser.add_argument("--diameter",  nargs='+', type=int,  default=[28])
parser.add_argument("--n_start", type=int,  default=0)
parser.add_argument("--n_end", type=int,  default=5000)
parser.add_argument(
    "--mask_percentage", type=int,   default=70)
parser.add_argument(
    "--offset", action="store_true", default=False, 
)

parser.add_argument(
    "--random_diameter", action="store_true", default=False, 
)

parser.add_argument(
    "--seed", type=int,   default=2024)
args = parser.parse_args()

diameter = args.diameter
mask_percentage = args.mask_percentage
offset = args.offset
random_diameter = args.random_diameter
shape = args.shape

save_dir = os.path.join(f'/workspace/radraid/projects/seg_lung/masked_seg/masking/{shape}_d{diameter}_p{mask_percentage}_offset{offset}_random{random_diameter}')
os.makedirs(save_dir, exist_ok = True)
np.random.seed(args.seed)

for i in range(args.n_start, args.n_end):
    if shape == 'cube':
        random_mask_patches_3d(i, img_size = (1,224,224,224), length=diameter, 
                          mask_percentage=mask_percentage,  offset = offset,
                         random_diameter = random_diameter)
    elif shape == 'sphere':
        random_mask_sphere_3d(i, img_size = (1,224,224,224), diameter=diameter, 
                            mask_percentage=mask_percentage,  offset = offset,
                            random_diameter = random_diameter)


