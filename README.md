# Enhancing Lung Segmentation Algorithms to Ensure Inclusion of Juxtapleural Nodules

[![IEEE Paper](https://img.shields.io/badge/IEEE-Paper-blue)](https://ieeexplore.ieee.org/abstract/document/10981085)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


**Luoting Zhuang, Seyed Mohammad Hossein Tabatabaei, Ashley Prosper, William Hsu**
Medical & Imaging Informatics, Department of Radiological Sciences, David Geffen School of Medicine at UCLA, Los Angeles, CA  

<p align="center">
    <img src="figures/overview.jpg"/> <br />
    <em> 
    Figure 1. An overview of the proposed algorithm and lung segmentation result on NLST test cases. 
    </em>
</p>

Computed tomography (CT) is pivotal in detecting and monitoring lung nodules in cancer screening. With the advancement of artificial intelligence in medical imaging, accurate lung segmentation has become crucial for reliable feature extraction. While traditional methods are not generalizable and computationally expensive, deep learning models also face difficulties incorporating nodules due to overreliance on pixel intensity. To overcome the challenge, we finetuned a 3D U-Net by randomly masking out 70% of the images, which forces the model to infer the missing regions and learn the boundaries of the lungs. Our model achieves a Dice coefficient of 0.982 in lung segmentation. Notably, our approach achieved higher sensitivity compared to three state-of-the-art deep learning models in the inclusion of juxtapleural and large nodules by 0.11, 0.20, and 0.52, respectively. Additionally, it consistently outperformed these models on external datasets. The improved result in nodule inclusion allows for more accurate and robust downstream analysis and computer-aided diagnosis of lung cancer. Our model also provides pixel-level uncertainty estimates, which visually present where the model is confident or uncertain. High-uncertainty areas can be flagged for further examination by both clinicians and researchers.

## Getting Started
### Create a Docker Container
```bash
docker run --shm-size=8g --gpus all -it --rm -v .:/workspace -v /etc/localtime:/etc/localtime:ro nvcr.io/nvidia/pytorch:23.05-py3
```
- If you use `-v .:/workspace` as shown above, Docker will map the **current directory** to `/workspace` inside the container.
- To map a different folder to a specific path in docker container, you can replace `-v .:/workspace` with `-v /path/to/local/folder:/path/in/container`.

### Clone the Repository and Install Packages
1. Go to the folder you want to store the code and clone the repo
```bash
git clone --depth 1 https://github.com/luotingzhuang/maskedSeg.git
cd maskedSeg
```

2. Install all of the required python packages using the following command line.
```bash
pip install -r requirements.txt
```

### Download Pretrained Weights
Download `model_weights` from the [link](https://drive.google.com/drive/folders/1elGnhviQBP8y7oPL2TpTn5jcBLE5HDs9?usp=drive_link) and put it under `./maskedSeg`.
- `checkpoint_final.pth` is the checkpoint file containing information about [totalsegmentator](https://github.com/wasserth/TotalSegmentator) architecture and its model weights.
- `es_checkpoint.pth.tar` is the weights of the finetuned model.
- `args.json` contains arguments for training.

```bash
# You can also download it using gdown
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1MiI7Vly9VtvxdTdDJ2PWIS--cgkVJjMv?usp=drive_link
```

### Data Requirement
The model accepts a **NIfTI** file as input and outputs either a **NIfTI** file or **NumPy** arrays.

To prepare a CSV file, list the path to the **NIfTI** file under the `image_path` column, along with the corresponding `pid`. 

The CSV file should contain two columns:  
| `pid` | `image_path` |  
|------|------------|  
| 001  | `./data/image1.nii.gz` |  
| 002  | `./data/image2.nii.gz` |  

Refer to `./dataset_csv/sample.csv` or `./dataset_csv/sample_paper.csv` as an example. `./dataset_csv/sample_paper.csv` contains three examples shown in Figure 2 in the manuscript.

Sample data can also be downloaded from the [link](https://drive.google.com/drive/folders/1elGnhviQBP8y7oPL2TpTn5jcBLE5HDs9?usp=drive_link).
```bash
# You can also download it using gdown
gdown --folder https://drive.google.com/drive/folders/1tQ_eD6i30C-qY9dfX4X20zuSyN7eB0lT?usp=drive_link
```
## Lung Segmentation
### Run Inference

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --csv_file ./dataset_csv/sample.csv --result_dir ./output --exp_dir ./model_weights --n 100 --thresh 0.55 --save_as nifti
```
#### Arguments
| Argument      | Type  | Default | Description |
|--------------|------|---------|-------------|
| `--csv_file`  | str  | `./dataset_csv/sample.csv` | Path to the CSV file containing image paths. |
| `--result_dir` | str  | `./output` | Directory to save the results. |
| `--exp_dir` | str  | `./model_weights` | Path to the experiment directory containing model weights. |
| `--n` | int | `100` | Number of masked samples to process. |
| `--thresh` | float | `0.55` | Threshold for segmentation. |
| `--save_as` | str | `nifti` | Output format (`nifti` or `numpy`). |

**Note:** The values `n=100` and `threshold=0.55` are used as the default values in the script. These parameters are also used to produce the results that are shown in the paper. You can adjust these values.

#### Outputs

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
  - The segmentation results are saved in RAS orientation.

#### Visualization
Use `./tutorial/visualization.ipynb` to visualize the the predicted segmentation in jupyter notebook.

### Other Lung Segmentation Tools
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [lungmask](https://github.com/JoHof/lungmask)

The segmentation results of sample data are stored in `baseline_sample_data_seg` on the [google drive](https://drive.google.com/drive/folders/1elGnhviQBP8y7oPL2TpTn5jcBLE5HDs9?usp=drive_link).

```bash
# You can also download it using gdown
gdown --folder https://drive.google.com/drive/folders/1S2k_cCIZV0SDLxRoQk9YL0x7mVTKgvr7?usp=drive_link
```

## Acknowledgements
This project is based on the code from the following repository:
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- [dynamic_network_architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures)

## TODO
- [ ] Training code

## CITATION
```bibtex
@inproceedings{zhuang2025enhancing,
  title={Enhancing Lung Segmentation Algorithms to Ensure Inclusion of Juxtapleural Nodules},
  author={Zhuang, Luoting and Tabatabaei, Seyed Mohammad Hossein and Prosper, Ashley E and Hsu, William},
  booktitle={2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
