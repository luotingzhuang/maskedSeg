# Enhancing Lung Segmentation Algorithms to Ensure Inclusion of Juxtapleural Nodules
**Luoting Zhuang, Seyed Mohammad Hossein Tabatabaei, Ashley Prosper, William Hsu**

Medical & Imaging Informatics, Department of Radiological Sciences, David Geffen School of Medicine at UCLA, Los Angeles, CA  

Computed tomography (CT) is pivotal in detecting and monitoring lung nodules in cancer screening. With the advancement of artificial intelligence in medical imaging, accurate lung segmentation has become crucial for reliable feature extraction. While traditional methods are not generalizable and computationally expensive, deep learning models also face difficulties incorporating nodules due to overreliance on pixel intensity. To overcome the challenge, we finetuned a 3D U-Net by randomly masking out 70% of the images, which forces the model to infer the missing regions and learn the boundaries of the lungs. Our model achieves a Dice coefficient of 0.982 in lung segmentation. Notably, our approach achieved higher sensitivity compared to three state-of-the-art deep learning models in the inclusion of juxtapleural and large nodules by 0.11, 0.20, and 0.52, respectively. Additionally, it consistently outperformed these models on external datasets. The improved result in nodule inclusion allows for more accurate and robust downstream analysis and computer-aided diagnosis of lung cancer. Our model also provides pixel-level uncertainty estimates, which visually present where the model is confident or uncertain. High-uncertainty areas can be flagged for further examination by both clinicians and researchers.

## Getting Started

### Package Requirement


### Data Requirement


## Lung Segmentation
### Run Inference
