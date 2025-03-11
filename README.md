# Enhancing Lung Segmentation Algorithms to Ensure Inclusion of Juxtapleural Nodules
**Luoting Zhuang, Seyed Mohammad Hossein Tabatabaei, Ashley Prosper, William Hsu**

Medical & Imaging Informatics, Department of Radiological Sciences, David Geffen School of Medicine at UCLA, Los Angeles, CA  

Computed tomography (CT) is pivotal in detecting and monitoring lung nodules in cancer screening. With the advancement of artificial intelligence in medical imaging, accurate lung segmentation has become crucial for reliable feature extraction. While traditional methods are not generalizable and computationally expensive, deep learning models also face difficulties incorporating nodules due to overreliance on pixel intensity. To overcome the challenge, we finetuned a 3D U-Net by randomly masking out 70% of the images, which forces the model to infer the missing regions and learn the boundaries of the lungs. Our model achieves a Dice coefficient of 0.982 in lung segmentation. Notably, our approach achieved higher sensitivity compared to three state-of-the-art deep learning models in the inclusion of juxtapleural and large nodules by 0.11, 0.20, and 0.52, respectively. Additionally, it consistently outperformed these models on external datasets. The improved result in nodule inclusion allows for more accurate and robust downstream analysis and computer-aided diagnosis of lung cancer. Our model also provides pixel-level uncertainty estimates, which visually present where the model is confident or uncertain. High-uncertainty areas can be flagged for further examination by both clinicians and researchers.

## Getting Started
### Clone the Repository and Download Weight
1. Clone the repo
```bash
git clone https://github.com/luotingzhuang/maskedSeg.git
cd maskedSeg
```
2. Download model weights from the [link](https://drive.google.com/drive/folders/1elGnhviQBP8y7oPL2TpTn5jcBLE5HDs9?usp=drive_link).
    - Download `checkpoint_final.pth` and save it to `./maskedSeg/totalsegmentator` folder.
        - This is the checkpoint file containing information about [totalsegmentator](https://github.com/wasserth/TotalSegmentator) architecture and its model weights.
    - Download the model_weights folder under `./maskedSeg`.
        - This is the weights of the finetuned model and arguments.

### Package Requirement


### Data Requirement
The model accepts a NIfTI file as input and outputs either a NIfTI file or NumPy arrays.
To prepare a CSV file, list the path to the NIfTI file under the `image_path` column, along with the corresponding `pid`. 
The CSV file should contain two columns:  
| `pid` | `image_path` |  
|------|------------|  
| 001  | `./data/image1.nii.gz` |  
| 002  | `./data/image2.nii.gz` |  

Refer to `./dataset_csv/sample.csv` as an example. The `seg_path` column is not required.
Sample data can also be downloaded from the [link](https://drive.google.com/drive/folders/1elGnhviQBP8y7oPL2TpTn5jcBLE5HDs9?usp=drive_link).

## Lung Segmentation
### Run Inference

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --csv_file ./dataset_csv/sample.csv --result_dir ./output --exp_dir ./model_weights --n 100 --thresh 0.55 --save_as nifti
```
**Arguments**  
| Argument      | Type  | Default | Description |
|--------------|------|---------|-------------|
| `--csv_file`  | str  | `./dataset_csv/sample.csv` | Path to the CSV file containing image paths. |
| `--result_dir` | str  | `./output` | Directory to save the results. |
| `--exp_dir` | str  | `./model_weights` | Path to the experiment directory containing model weights. |
| `--n` | int | `100` | Number of masked samples to process. |
| `--thresh` | float | `0.55` | Threshold for segmentation. |
| `--save_as` | str | `nifti` | Output format (`nifti` or `numpy`). |

**Note:** The values `n=100` and `threshold=0.55` are used as the default values in the script. These parameters are also used to produce the results that are shown in the paper. You can adjust these values.

**Outputs**
By default, the segmentation results will be saved in the `./output` folder.  

- If saved as **NIfTI** files (`--save_as nifti`):  
  - A separate folder will be created for each `pid`.  
  - Inside each folder, the following files will be saved:  
    - `mean.nii.gz` – The average segmentation result across n masked samples.  
    - `std.nii.gz` – The standard deviation of the segmentation.  
  - Saving as **NIfTI** files may take longer.

- If saved as **NumPy** files (`--save_as numpy`):  
  - A single `.npz` file will be saved for each `pid` in the output directory.  
  - The file format will be `{pid}.npz`, containing:  
    - **First array** – The average segmentation result across n masked samples.  
    - **Second array** – The standard deviation.  