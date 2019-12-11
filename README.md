# HybridPose: 6D Object Pose Estimation under Hybrid Representations
This repository contains authors' implementation of [HybridPose: 6D Object Pose Estimation under Hybrid Representations](http://songc.me). Our implementation is based on [PVNet](https://github.com/zju3dv/pvnet).
We warmly welcome any discussions related to our implementation and our paper. Please feel free to open an issue.

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

They are uploaded here: [Linemod](https://drive.google.com/open?id=1Ml3SZMe8ZG6GXPZzgsa7bQNYVCeAvkhM), [Occlusion Linemod](https://drive.google.com/open?id=1npaDv5GqPljbLE3_rry0KRl8lJwyS2I8).

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
* Linemod:
* Occlusion Linemod: [ape](https://drive.google.com/open?id=1mSMJ_PuFJZ9heOG2NbR18pEYmXCuiNm5)

We have configured random seeds in src/train\_core.py and expect you to re-produce identical weights by running our training script. Our training uses two graphics cards with a batch size of 12.

After you download the pre-trained weights, unzip them somewhere and configure `--load_dir` in src/train\_core.py to the unzipped weights (e.g. saved\_weights/occlusion\_linemod/ape/checkpoints/0.02/499).

Running `src/train_core.py` now will save both ground truth and predicted poses to a directory called `output`.
