from skimage import measure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


class MyArgs:
    def __init__(self, **kwargs):
        # Assign all keyword arguments to attributes
        self.__dict__.update(kwargs)
        self.sche = None
        self.max_epoch = 50

def return_nodule(img, x, y, z, r_x, r_y, r_z):
    return img[x - r_x : x + r_x, y - r_y : y + r_y, z - r_z : z + r_z]





def find_connected_components(image, size_threshold):
    # Convert the image to binary using a suitable threshold
    binary_image = image > 0.5  # Adjust the threshold as needed

    # Label connected components
    labeled_image = measure.label(binary_image)

    # Measure properties of labeled regions
    regions = measure.regionprops(labeled_image)

    # Filter components based on size threshold
    large_components = [region for region in regions if region.area > size_threshold]

    return labeled_image, large_components



def get_dice_coefficient(real_mask, pred_mask):
    intersection = (real_mask * pred_mask).sum()
    union = real_mask.sum() + pred_mask.sum()
    return 2 * intersection / union


def get_sensitivity(real_mask, pred_mask):
    return (real_mask * pred_mask).sum() / real_mask.sum()


def get_precision(real_mask, pred_mask):
    return (real_mask * pred_mask).sum() / pred_mask.sum()


def get_metrics(
    thresh,
    corrected_seg,
    mean_seg,
    # majority_seg,
    lungmask_seg,
    total_seg,
    corrected_seg_nodule,
    mean_seg_nodule,
    # majority_seg_nodule,
    lungmask_seg_nodule,
    total_seg_nodule,
):
    dice_nodule = pd.DataFrame(
        {
            "metric": "dice_nodule",
            "thresh": thresh,
            "model_mean": [get_dice_coefficient(corrected_seg_nodule, mean_seg_nodule)],
            #'model_majority': [get_dice_coefficient(corrected_seg_nodule,majority_seg_nodule)],
            "lungmask": [
                get_dice_coefficient(corrected_seg_nodule, lungmask_seg_nodule)
            ],
            "totalseg": [get_dice_coefficient(corrected_seg_nodule, total_seg_nodule)],
        }
    )

    sensitivity_nodule = pd.DataFrame(
        {
            "metric": "sensitivity_nodule",
            "thresh": thresh,
            "model_mean": [get_sensitivity(corrected_seg_nodule, mean_seg_nodule)],
            #'model_majority': [get_sensitivity(corrected_seg_nodule,majority_seg_nodule)],
            "lungmask": [get_sensitivity(corrected_seg_nodule, lungmask_seg_nodule)],
            "totalseg": [get_sensitivity(corrected_seg_nodule, total_seg_nodule)],
        }
    )

    precision_nodule = pd.DataFrame(
        {
            "metric": "precision_nodule",
            "thresh": thresh,
            "model_mean": [get_precision(corrected_seg_nodule, mean_seg_nodule)],
            #'model_majority': [get_precision(corrected_seg_nodule,majority_seg_nodule)],
            "lungmask": [get_precision(corrected_seg_nodule, lungmask_seg_nodule)],
            "totalseg": [get_precision(corrected_seg_nodule, total_seg_nodule)],
        }
    )

    dice_lung = pd.DataFrame(
        {
            "metric": "dice_lung",
            "thresh": thresh,
            "model_mean": [get_dice_coefficient(corrected_seg, mean_seg)],
            #'model_majority': [get_dice_coefficient(corrected_seg,majority_seg)],
            "lungmask": [get_dice_coefficient(corrected_seg, lungmask_seg)],
            "totalseg": [get_dice_coefficient(corrected_seg, total_seg)],
        }
    )

    return pd.concat([dice_lung, dice_nodule, sensitivity_nodule, precision_nodule])

def plot_overlay(arrs):
    fig, ax = plt.subplots(1,len(arrs) , figsize = (len(arrs) * 5, 10))
    for i, (title, arr) in enumerate(arrs.items()):
        try:
        
            img = arr['img']

            # Display the grayscale image
            ax[i].imshow(img, cmap='gray')
        except:
            seg = arr['seg']
            ax[i].imshow(seg, cmap='jet',vmin = 0,vmax = arr['vmax'])
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(title)
            continue
        try:
            seg = arr['seg']
            segmentation_rgba = np.zeros([i for i in seg.shape] +[4])
            segmentation_rgba[:, :, 0] = seg
            segmentation_rgba[:, :, 3] = 0.5 

            # Overlay the segmentation mask with transparency
            ax[i].imshow(segmentation_rgba, cmap='jet', alpha=0.5)
        except:
            pass

        # Remove axis ticks
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(title)

    plt.show()

def load_img(path):
    sitk_img = sitk.ReadImage(path)
    sit_arr = sitk.GetArrayFromImage(sitk_img)
    return sitk_img, sit_arr

#plot y1_img, y1_seg, pred_seg

def plot_img(arrs):
    fig, ax = plt.subplots(1,len(arrs) , figsize = (len(arrs) * 5, 10))
    for i, (title, arr) in enumerate(arrs.items()):
        ax[i].imshow(arr,cmap = 'gray')
        ax[i].set_xticks([]) 
        ax[i].set_yticks([]) 
        ax[i].set_title(title)
        
    plt.show()

def reori(data):
    return np.rot90(data,k = 1)
