# <p align="center">Topology-Preserving Computed Tomography Super-resolution Based on Dual-stream Diffusion Model</p>
Here is the project response MICCAI 2023: "Topology-Preserving Computed Tomography Super-resolution Based on Dual-stream Diffusion Model". If you encounter any questions, please feel free to contact us. You can create an issue or just send an email to me at yuetan.chu@kaust.edu.sa. Also welcome for any idea exchange and discussion.

## Abstract
![image](https://github.com/Arturia-Pendragon-Iris/UHRCT_SR/blob/main/figure/network%20architecture.png)
X-ray computed tomography (CT) is indispensable for modern medical diagnosis, but the degradation of spatial resolution and image quality can adversely affect analysis and diagnosis. Although super-resolution (SR) techniques can help restore lost spatial information and improve imaging resolution for low-resolution CT (LRCT), they are always criticized for topology distortions and secondary artifacts. To address this challenge, we propose a dual-stream diffusion model for super-resolution with topology preservation and structure fidelity. The diffusion model employs a dual-stream structure-preserving network and an imaging enhancement operator in the denoising process for image information and structural feature recovery. The imaging enhancement operator can achieve simultaneous enhancement of vascular and blob structures in CT scans, providing the structure priors in the super-resolution process. The final super-resolved CT is optimized in both the convolutional imaging domain and the proposed vascular structure domain. Furthermore, for the first time, we constructed an ultra-high resolution CT scan dataset with a spatial resolution of $0.34\times0.34$ $mm^2$ and an image size of $1024\times 1024$ as a super-resolution training set. Quantitative and qualitative evaluations show that our proposed model can achieve comparable information recovery and much better structure fidelity compared to the other state-of-the-art methods. The performance of high-level tasks, including vascular segmentation and lesion detection on super-resolved CT scans, is comparable to or even better than that of raw HRCT.


## Dataset
Here we share the ultra-high-resolution (UHRCT) CT scans (1024*1024) with 1.00 mm inter-slice thickness. The link to the dataset is [here](https://drive.google.com/drive/folders/1EE0oLV-PuYxMdSuCcXPE9qw5xo8cw4gi?usp=sharing). The shared dataset consists of 95 URCT datasets in the format of .npz. You can use np.load(path_to_your_dataset)["arr_0"] to open them. Some of the CT scans are clipped to [-1000, 600].

## Code
We have uploaded the code for the implementation of the conv filter, the UNet architecture, and the loss function we use in our study. For the training processing, we follow the code released by [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement).

## Results
![image](https://github.com/Arturia-Pendragon-Iris/UHRCT_SR/blob/main/figure/SR_results.png)
![image](https://github.com/Arturia-Pendragon-Iris/UHRCT_SR/blob/main/figure/downstream_tasks.png)

## Acknowledgments
This publication is based upon work supported by the King Abdullah University of Science and Technology (KAUST) Office of Research Administration (ORA) under Award No URF/1/4352-01-01, FCC/1/1976-44-01, FCC/1/1976-45-01, REI/1/5234-01-01, REI/1/5414-01-01.

## Cite
If you use the dataset or refer to our paper, please use the citation as follows:
```
@inproceedings{chu2023topology,
  title={Topology-Preserving Computed Tomography Super-Resolution Based on Dual-Stream Diffusion Model},
  author={Chu, Yuetan and Zhou, Longxi and Luo, Gongning and Qiu, Zhaowen and Gao, Xin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={260--270},
  year={2023},
  organization={Springer}
}
```

