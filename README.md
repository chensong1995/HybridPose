# HybridPose: 6D Object Pose Estimation under Hybrid Representations
This repository contains authors' implementation of [HybridPose: 6D Object Pose Estimation under Hybrid Representations](https://arxiv.org/abs/2001.01869). Our implementation is based on [PVNet](https://github.com/zju3dv/pvnet).
We warmly welcome any discussions related to our implementation and our paper. Please feel free to open an issue.

**News (October 16, 2020):** We have updated our experiments using the conventional data split on Linemod/Occlusion Linemod. Following baseline works, we use around 15% of Linemod examples for training. The rest of Linemod examples, as well as the entire Occlusion Linemod dataset, are used for testing. Both this GitHub repository and the arXiv paper are updated. HybridPose achieves an ADD(-S) score of 0.9125577238 on Linemod, and 0.4754330537 on Occlusion Linemod. We sincerely appreciate the readers who pointed out this issue to us, including but not limited to [Shun Iwase](https://github.com/sh8) and [hiyyg](https://github.com/hiyyg).

## Introduction
HybridPose consists of intermediate representation prediction networks and a pose regression module. The prediction networks take an image as input, and output predicted keypoints, edge vectors, and symmetry correspondences. The pose regression module consists of a initialization sub-module and a refinement sub-module. The initialization sub-module solves a linear system with predicted intermediate representations to obtain an initial pose. The refinement sub-module utilizes GM robust norm to obtain the final pose prediction.
![Approach overview](./assets/overview.png)

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
* Keypoints: both 2D and 3D coordinates. These labels are generated using [FSP](https://github.com/zju3dv/pvnet/blob/master/lib/utils/data_utils.py).
* Symmetry: Symmetry correspondences in 2D and the normal of symmetry plane in 3D. These labels are generated using [SymSeg](https://github.com/aecins/symseg).
* Segmentation masks: On Linemod, we create segmentation masks by projecting 3D models.

They are uploaded here:
* OneDrive: [Linemod](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/EX1-jCyvyw9DoNGDGRY6RXsBTYV0g0n-4YZjGc4F_9xTRA?e=qmV7bb), [Occlusion Linemod](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/EaHVyE8WaVFJh5tdtJIPxXgBo1iCIFeCXJHdFQxXT3NxyA?e=yNdWKe).
* Tencent Weiyun: [Linemod](https://share.weiyun.com/VOf5yOZI), [Occlusion Linemod](https://share.weiyun.com/50i7KTb).

The following commands unzip these labels to the correct directory:
```
unzip data/temp/linemod_labels.zip -d data/linemod
unzip data/temp/occlusion_labels.zip -d data/occlusion_linemod
```

We also use the [synthetic data from PVNet](https://github.com/zju3dv/pvnet-rendering/). Please generate blender rendering and fuse data using their code.  After data generation, please place blender data in `data/blender_linemod`, and fuse data in `data/fuse_linemod`. The directory structure should look like this:

```
data
  |-- blender_linemod
  |         |---------- ape
  |         |---------- benchviseblue
  |         |---------- cam
  |         |---------- ... (other objects)
  |-- fuse_linemod
  |         |---------- fuse
  |         |            |---------- 0_info.pkl
  |         |            |---------- 0_mask.png
  |         |            |---------- 0_rgb.jpg
  |         |            |---------- 1_info.pkl
  |         |            |---------- 1_mask.png
  |         |            |---------- 1_rgb.jpg
  |         |            |---------- ... (other examples)
```

After that, please use `data/label.py` and `data/label_fuse.py` to create intermediate representation labels blender and fuse data, respectively.

```
mv data/label.py data/blender_linemod/label.py
cd data/blender_linemod
python label.py
```

```
mv data/label_fuse.py data/fuse_linemod/label_fuse.py
cd data/fuse_linemod
python label_fuse.py
```

One of the arguments taken by the labeling scripts is `--pvnet_linemod_path`. This is the data directory used by PVNet render. The structure of this directory looks like this:

```
pvnet_lindmod
 |-- ape
 |    |--- amodal_mask
 |    |--- contours
 |    |--- JPEGImages
 |    |--- labels
 |    |--- labels_occlusion
 |    |--- mask
 |    |--- nosiy_contours
 |    |--- pose
 |    |--- ape.ply
 |    |--- ... (several other .txt and .pkl files)
 |-- benchviseblue
 |    |--- ...
 |-- cam
 |    |--- ...
 |-- ... (other objects)
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
You can download our pre-trained weights below. We use train one set of weights on Linemod, and test on both Linemod and Occlusion Linemod:
* OneDrive: [ape](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/Ec9RFKrnmi5Gp6mpG0rzemcBTf_UG8L-Jys3T9fqOQ1r2Q?e=1StQkC),
[benchviseblue](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/EThQc5k0ToVIpXzTl2dSsy8BLgpgQj4Hr7Z9-heJCUAGNg?e=OGviAL),
[cam](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/EXzddKdkvJpBt-asNlfRJssBD08guI7IAmKAT00DghGt9A?e=5UgEG6),
[can](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/ERrsKeRlwehPjpl4RTV2muUB7AnrRtYl5mqmUYO61NdAkg?e=DVYzoO),
[cat](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/EbsJ9ij5JthPj9Il7Ms3C_gBYoaJDSS8jVEXJnikLde7nA?e=KcQ5g4),
[driller](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/EVi5qRlJV-1Fn2mQlR-GVwABZ1_JHD0pOIA3YBu6DQ-kdg?e=6fGsAY),
[duck](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/EbpYzIDL8UtMkqWwY8RSaC8B8-hfTrkWxtQoRWt--GCaWw?e=QFDUSo),
[eggbox](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/ERGay7Ell5NMnmgSGWTU0WUBBPJLU8gU3V0y2tU9CNZ1Ag?e=EOAuYJ),
[glue](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/EUWc7032Kl9MtvynSeEnPeoB4gt4iqmyUj2dwBhO0KZIWQ?e=wVynHT),
[holepuncher](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/Eev3foHkKpdKveetchoNjE0BVdOoJ4Zs7rCfl_Ea8nkvrA?e=7TTlsC),
[iron](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/EX6xS32n2oNHsCW49-q7uJ0B9lYd9FdHe6kx6DexY4LMoQ?e=S64eDF),
[lamp](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/EY5wJyvD4AFPqzbYUOTfgXgBb_z9qiigypgSLcRvJpDvBg?e=g9PJdf),
[phone](https://utexas-my.sharepoint.com/:u:/g/personal/song_austin_utexas_edu/EdKEv70zs8JGuwaJfRYKwrYBhr_bG4Ig0PJCf2oZmNoJ_Q?e=q9oZhL)
* Tencent Weiyun: [ape](https://share.weiyun.com/yOCM20YC),
[benchviseblue](https://share.weiyun.com/iNCkC7iN),
[cam](https://share.weiyun.com/4jE1JxQK),
[can](https://share.weiyun.com/HuVBksHq),
[cat](https://share.weiyun.com/WIAUu2kc),
[driller](https://share.weiyun.com/oYfPFsj6),
[duck](https://share.weiyun.com/5liVTjld),
[eggbox](https://share.weiyun.com/CjQyLtbt),
[glue](https://share.weiyun.com/Xq7IlKf4),
[holepuncher](https://share.weiyun.com/t2eA816n),
[iron](https://share.weiyun.com/cPzMB2Rx),
[lamp](https://share.weiyun.com/W5YBK8UA),
[phone](https://share.weiyun.com/4gDBsjls)

~We have configured random seeds in src/train\_core.py and expect you to re-produce identical weights by running our training script.~ It turns out that [completely reproducible results are not guaranteed across PyTorch releases, individual commits or different platforms. Furthermore, results need not be reproducible between CPU and GPU executions, even when using identical seeds.](https://pytorch.org/docs/stable/notes/randomness.html) Also, the randomness in the PVNet synthetic data generation will create some difference in training outcome. Our training uses two graphics cards with a batch size of 10.

After you download the pre-trained weights, unzip them somewhere and configure `--load_dir` in `src/train_core.py` to the unzipped weights (e.g. `saved_weights/linemod/ape/checkpoints/0.001/199`).

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
