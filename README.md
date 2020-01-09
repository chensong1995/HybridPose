# HybridPose: 6D Object Pose Estimation under Hybrid Representations
This repository contains authors' implementation of [HybridPose: 6D Object Pose Estimation under Hybrid Representations](https://arxiv.org/abs/2001.01869). Our implementation is based on [PVNet](https://github.com/zju3dv/pvnet).
We warmly welcome any discussions related to our implementation and our paper. Please feel free to open an issue.

## Introduction
We introduce HybridPose, a novel 6D object pose estimation approach. HybridPose utilizes a hybrid intermediate representation to express different geometric information in the input image, including keypoints, edge vectors, and symmetry correspondences. Compared to a unitary representation, our hybrid representation allows pose regression to exploit more and diverse features when one type of predicted representation is inaccurate (e.g., because of occlusion). HybridPose leverages a robust regression module to filter out outliers in predicted intermediate representation. We show the robustness of HybridPose by demonstrating that all intermediate representations can be predicted by the same simple neural network without sacrificing the overall performance. Compared to state-of-the-art pose estimation approaches, HybridPose is comparable in running time and is significantly more accurate. For example, on Occlusion Linemod dataset, our method achieves a prediction speed of 30 fps with a mean ADD(-S) accuracy of 79.2%, representing a 67.4% improvement from the current state-of-the-art approach.

![Approach overview](./assets/overview.png)

HybridPose consists of intermediate representation prediction networks and a pose regression module. The prediction networks take an image as input, and output predicted keypoints, edge vectors, and symmetry correspondences. The pose regression module consists of a initialization sub-module and a refinement sub-module. The initialization sub-module solves a linear system with predicted intermediate representations to obtain an initial pose. The refinement sub-module utilizes GM robust norm to obtain the final pose prediction.

## Download
```
git clone --recurse-submodules git@github.com:chensong1995/HybridPose.git
```

## Environment set-up
Please install [Anaconda](https://www.anaconda.com/distribution/) first and execute the following commands:
```
conda create -y --name hybridpose python==3.7.4
conda install -y -q --name hybridpose -c pytorch -c anaconda -c conda-forge -c pypi --file requirements.txt
conda activate hybridpose
```

## Compile the Ransac Voting Layer
The Ransac Voting Layer is used to generate keypoint coordinates from vector fields. Please execute the following commands (copied from [PVNet](https://github.com/zju3dv/pvnet)):
```
cd lib/ransac_voting_gpu_layer
python setup.py build_ext --inplace
```

## Compile the pose regressor
The pose regressor is written in C++ and has a Python wrapper. Please execute the following commands:
```
cd lib/regressor
make
```

## Dataset set-up
We experimented HybridPose on Linemod and Occlusion Linemod. Let us first download the original datasets using the following commands:
```
python data/download_linemod.py
python data/download_occlusion.py
```
Let us then download our augumented labels to these two datasets. Our augumented labels include:
* Blender meshes on Linemod objects: For some reasons, pose labels on Linemod are not aligned perfectly with the 3D models. After discussions with the authors of [PVNet](https://github.com/zju3dv/pvnet), we followed their advice and used Blender meshes to correct Linemod pose labels.
* Keypoints: both 2D and 3D coordinates. These labels are generated using [FSP](https://github.com/zju3dv/pvnet/blob/master/lib/utils/data_utils.py).
* Symmetry: Symmetry correspondences in 2D and the normal of symmetry plane in 3D. These labels are generated using [SymSeg](https://github.com/aecins/symseg).
* Segmentation masks: On Linemod, we create segmentation masks by projecting 3D models. On Occlusion Linemod, we use the segmentation masks provided in [PVNet](https://github.com/zju3dv/pvnet).

They are uploaded here: [Linemod](https://drive.google.com/file/d/1f9-KEVtKprU0vNYWXjPSFhEoU32Vtlv2/view?usp=sharing), 
[Occlusion Linemod](https://drive.google.com/file/d/1PItmDj7Go0OBnC1Lkvagz3RRB9qdJUIG/view?usp=sharing).

The following commands unzip these labels to the correct directory:
```
unzip data/temp/linemod_labels.zip -d data/linemod
unzip data/temp/occlusion_labels.zip -d data/occlusion_linemod
```

## Training
Please set the arguments in src/train\_core.py execute the following command (note that we need to set LD\_LIBRARY\_PATH for the pose regressor):
```
# on bash shell
LD_LIBRARY_PATH=lib/regressor:$LD_LIBRARY_PATH python src/train_core.py
# on fish shell
env LD_LIBRARY_PATH="lib/regressor:$LD_LIBRARY_PATH" python src/train_core.py
```
If you use a different shell other than bash and fish, prepend "lib/regressor" to LD\_LIBRARY\_PATH and run `python src/train_core.py`.

## Pre-trained weights
You can download our pre-trained weights below:
* Linemod: [ape](https://drive.google.com/file/d/19Nl8AOER9brGDGUGu1WRwhdFBJNLymiu/view?usp=sharing),
[benchviseblue](https://drive.google.com/file/d/1nMLJtV3XsK60bGGE-zFA1dw34074yryf/view?usp=sharing),
[cam](https://drive.google.com/file/d/1Sc0wx73E_DyrKe1N7DMl3qIRKSimIZoe/view?usp=sharing),
[can](https://drive.google.com/file/d/1NTEc6BcTV69Li0XW-ZDD7aLMuIkY3RL5/view?usp=sharing),
[cat](https://drive.google.com/file/d/1DN5OULGOtVP7r8hNySl2Ufou_tLSdWpI/view?usp=sharing),
[driller](https://drive.google.com/file/d/1JFiBxbp6nSKnsDsJK2II0WUYRi5oYdQF/view?usp=sharing),
[duck](https://drive.google.com/file/d/1XlVV1CBrPxZgZwNc9EjqGfTO1XSa2DhH/view?usp=sharing),
[eggbox](https://drive.google.com/file/d/1KyVC_sU0H8-VXjSz0yCAuyOjL-Olfylq/view?usp=sharing),
[glue](https://drive.google.com/file/d/1ZU5V4ew97XbzCmQ94mQhLCQtH5GttZ_i/view?usp=sharing),
[holepuncher](https://drive.google.com/file/d/1BVlQTmQOxs4pYjEI19eunA5BO-Q6AiAA/view?usp=sharing),
[iron](https://drive.google.com/file/d/1CtZfFycD90xcu3u6dEjoQa0ETY0RSJ5V/view?usp=sharing),
[lamp](https://drive.google.com/file/d/1UYnDxdXs_XVNyz7QHeq3RIPU1Gw-df-r/view?usp=sharing),
[phone](https://drive.google.com/file/d/1ArP9c7Z-CG2P9zvhreA4_jj0-e0i1TSF/view?usp=sharing)
* Occlusion Linemod: [ape](https://drive.google.com/file/d/1JeBETMGgELrawzofO59j4OCpg-2tf3iy/view?usp=sharing),
[can](https://drive.google.com/file/d/1Cl47bGiPyodHNqITaxCadFAT97YP7nl9/view?usp=sharing),
[cat](https://drive.google.com/file/d/1gDMwqPuFyKg_YW_PbqY_yT53dJEYYrqW/view?usp=sharing),
[driller](https://drive.google.com/file/d/1iAvptsTtwHVp6bNNSRBl5QiVi3O8uDeo/view?usp=sharing),
[duck](https://drive.google.com/file/d/1GwmhyWG4czIsVcCRyWA19ZEZfTzEN2Wo/view?usp=sharing),
[eggbox](https://drive.google.com/file/d/1UKl6aSLRVZzbjI1b5yhxBRlavI8n_JMb/view?usp=sharing),
[glue](https://drive.google.com/file/d/1JnABWWuNns_syYO-zPUBGViT_HWt0VAW/view?usp=sharing),
[holepuncher](https://drive.google.com/file/d/1XGt5BvYEbVN67zZbdMaGBsaC2-pad4zv/view?usp=sharing)

We have configured random seeds in src/train\_core.py and expect you to re-produce identical weights by running our training script. Our training uses two graphics cards with a batch size of 12.

After you download the pre-trained weights, unzip them somewhere and configure `--load_dir` in `src/train_core.py` to the unzipped weights (e.g. `saved_weights/occlusion_linemod/ape/checkpoints/0.02/499`).

Running `src/train_core.py` now will save both ground truth and predicted poses to a directory called `output`.

## Evaluation
To evaluate ADD(-S) accuracy of predicted poses, please set the arguments in `src/evaluate.py` and run
```
python src/evaluate.py
```

## Citation
If you find our work useful in your research, please kindly make a citation using:
```
@misc{song2020hybridpose,
    title={HybridPose: 6D Object Pose Estimation under Hybrid Representations},
    author={Chen Song and Jiaru Song and Qixing Huang},
    year={2020},
    eprint={2001.01869},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
