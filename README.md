nbv_3d_cnn_octree
=================

2024-05-28

This repository includes the nbv_3d_cnn_octree ROS package, which was used for octree-based occupancy completion for next best view (NBV) planning. The package contains implementations of a sparse octree-based CNN which predicts the environment occupancy probability given a partially complete environment with occupied, empty, and unknown voxels.

The repository also includes the nbv_3d_cnn package. This package was originally published in the [nbv_3d_cnn_prob](https://github.com/RMonica/nbv_3d_prob_cnn) repository to evaluate multiple NBV methods. The version of nbv_3d_cnn in this repository was extended to include the `AutocompleteOctreeIGain` and `AutocompleteVoxelgridIGain` methods, which use the new octree-based occupancy completion. The original README for nbv_3d_cnn_prob is available [here](./README_nbv_3d_cnn.md).

**Reference publication:**

- R. Monica and J. Aleotti, "A Sparse Octree-Based CNN for Probabilistic Occupancy Prediction Applied to Next Best View Planning", in IEEE Robotics and Automation Letters, doi: 10.1109/LRA.2024.3460432  
  <https://ieeexplore.ieee.org/document/10679903>

Installation
------------

These are ROS packages. Place them into your workspace and build them using catkin (*catkin tools* is preferred over catkin_make).

**Note**: by default, ROS compiles packages without compiler optimizations. Enable optimizations for better performance.

The software was tested on Ubuntu 20.04.

**Dependencies**

- ROS (Robot Operating System) noetic
- PCL (Point Cloud Library)
- OpenCV
- OpenCL
- Eigen3
- OctoMap
- PyTorch
- [rmonica_voxelgrid_common](https://github.com/RMonica/rmonica_voxelgrid_common)
- Minkowski Engine (optional, see below)

**Minkowski Engine**

The [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) is an engine for accelerating sparse tensor convolution in PyTorch. This dependency is optional. The Minkowski Engine is used only if the argument `engine` is set to `minkowski` instead of `pytorch` in the launch files.

Please download the source code from the Minkowski Engine [github repository](https://github.com/NVIDIA/MinkowskiEngine). You may want to apply the patch `deps/minkowski.patch` to fix a few issues.

The Minkowski Engine must be built and installed where it can be found by Python. In my installation, it was installed in the folder `nbv_3d_cnn_octree/deps/MinkowskiEngine_install`. Edit the `PYTHONPATH` environment variable defined in the launch files to point to the correct location in your system.

**Note**: some code in this repository still references [TorchSparse](https://github.com/mit-han-lab/torchsparse), another sparse tensor engine. I was unable to make this engine work for some reason. The code still attempts to load the engine if present, but it will not be used.

Usage
-----

### 2D Dataset

- Place the 2D scene dataset, composed of `.tif` files, into `nbv_3d_cnn/data/inria_dataset/AerialImageDataset/train/gt/`. Files in the correct format may be downloaded from the ground truth of the [INRIA](https://project.inria.fr/aerialimagelabeling/) dataset. The dataset is very large, but only the ground truth (the black and white images) is needed.
- Create folder `nbv_3d_cnn/data/inria_environments` if not existing.
- Launch `nbv_3d_cnn_octree/launch/dataset/generate_test_dataset.launch` to create the dataset images.
- Create folder `nbv_3d_cnn_octree/data/inria_environments` if not existing.
- Launch `nbv_3d_cnn_octree/launch/generate_octrees_2d.launch` to convert the images into quadtrees (2D octrees).

### 3D Dataset

- Place the 3D scene dataset into `nbv_3d_cnn/data/scenes_realistic`. These are voxel grids with 128×128×96 resolution, represented as OctoMap binary octrees. Example datasets can be downloaded from [here](http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_octree_scenes_realistic.zip) or [here](http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_scenes_realistic.zip).
- Create folder `nbv_3d_cnn/data/environments_3d_realistic` if not existing.
- Launch `nbv_3d_cnn_octree/launch/dataset/generate_test_dataset_3d_realistic.launch`
- Create folder `nbv_3d_cnn_octree/data/environments_3d_realistic` if not existing.
- Launch `nbv_3d_cnn_octree/launch/dataset/generate_octrees_3d.launch` to convert the images into octrees.

### Training

- For 2D, use the launch file `nbv_3d_cnn_octree/launch/octree_sparse.launch`. For 3D, use the launch file `nbv_3d_cnn_octree/launch/octree_sparse_3d.launch`.
- Configure the launch file. Set the arguments:
    - **model_type**: `enc_dec` for dense encoder-decoder, `sparse_enc_dec` for sparse encoder-decoder, `resnet` for dense encoder-decoder with resblocks, `sparse_resnet` for sparse encoder-decoder with resblocks.
    - **outdir**: an existing folder for the output data. In 2D, it is a path relative to `nbv_3d_cnn_octree/data/inria_environments`. In 3D, it is relative to `nbv_3d_cnn_octree/data/environments_3d_realistic`. Create a new folder if necessary.
    - **engine**: `pytorch` or `minkowski`.
    - **unified_loss**: if true, it uses the multi-scale loss, otherwise it uses the Structure+Task loss.
- Launch the launch file.

**Note**: if **unified_loss** is enabled, training continues from the last checkpoint of the corresponding training with the Structure+Task loss. See how the parameter **load_checkpoint_file** is used in the launch file.

### Next best view simulation

A next best view exploration task can be executed in simulation using the trained networks and a probabilistic NBV planner. Example launch files are provided:

- `nbv_3d_cnn_octree/launch/simulate_nbv_cycle.launch` for 2D environments (images). Example scenes (cropped from the INRIA dataset) can be downloaded from [here](http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_octree_inria_environments_400.zip).
- `nbv_3d_cnn_octree/launch/simulate_nbv_cycle_3d_realistic.launch` for 3D environments, 128×128×96 resolution. Scenes from training dataset can be used (see above for links).
- `nbv_3d_cnn_octree/launch/simulate_nbv_cycle_3d_realistic_large.launch` for 3D environments, 256×256×192 resolution. Example scenes can be downloaded from [here](http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_octree_scenes_realistic_large.zip).

Each launch files starts a `simulate_nbv_cycle` C++ node, which manages the NBV execution. It also starts an `octree_predict.py` node for each network that may be used for occupancy prediction. Only one network will be actually used, as configured by the **method** argument.

Each network node must be configured with the same parameters used for training, and the **checkpoint_file** parameter of each node must point at the file with the trained network parameters. The convention used in the launch files is that the node are named `nbv_3d_cnn_$(arg method)`, where **method** is the name of the network.

Pre-trained network parameters may be downloaded from [here (2D)](http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_octree_pre_trained_2d.zip) and [here (3D)](http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_octree_pre_trained_3d.zip). With the default configuration, these archives should be extracted into the folders `nbv_3d_cnn_octree/data/inria_environments` and `nbv_3d_cnn_octree/data/environments_3d_realistic`, respectively.

**Parameters and arguments**

Relevant arguments of the launch file are:

- **method**: the name of the network for occupancy prediction. The `simulate_nbv_cycle` node will look for an action named `predict` in the namespace of the node named `nbv_3d_cnn_$(arg method)`. These networks are defined in the launch file:
    - `enc_dec`: dense encoder-decoder.
    - `sparse_enc_dec_st_loss`: sparse encoder-decoder trained with Structure+Task loss.
    - `sparse_enc_dec_leak05_a09`: sparse encoder-decoder trained with the multi-scale loss.
- **use_octree_for_prediction**: uses the **method** argument to determine if the network expects an octree or a voxel grid as input.
- **use_octree_for_nbv**: whether to use an octree for ray casting in the probabilistic NBV method. A voxel grid is used otherwise. This is orthogonal to the network selection: if the network uses octrees and NBV does not (or *vice versa*), conversions will be carried out.
- **image_file_name**: name of the input environment file. By default, this is built from the **image_index** argument.
- **outdir**: path to save the output files. This value is used to build a path relative to folders `nbv_3d_cnn_octree/data/simulate_nbv_cycle` (2D), `nbv_3d_cnn_octree/data/simulate_nbv_cycle_3d_realistic` (3D), `nbv_3d_cnn_octree/data/simulate_nbv_cycle_3d_realistic_large` (3D large).
- **save_images**: if true, debug images/voxelgrids and data are saved in addition to statistics.

Relevant parameters in the `simulate_nbv_cycle` node may be also:

- **max_iterations**: number of NBV iterations.
- **nbv_algorithm**: it may be `Random` (selects NBV at random), `OmniscientGain` (selects NBV using the ground truth), `AutocompleteOctreeIGain` (octree-based NBV) and `AutocompleteVoxelgridIGain` (voxelgrid-based NBV).
- **sample_fixed_number_of_views**: number of viewpoints which are sampled for NBV.
- **igain_min_range**, **sensor_range_voxels**: minimum and maximum virtual sensor range (in number of cells).
- **sensor_resolution_y**, **sensor_resolution_x**, **sensor_focal_length**: virtual sensor parameters.

2024-09-20