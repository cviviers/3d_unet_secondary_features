# Pancreatic Ductal Adenocarcinoma Segmentaion using a 3D U-Net and Utilizing Secondary Tumour-Indicative Features

Resources shared as part the papers:
1. [Improved Pancreatic Tumor Detection by Utilizing Clinically-Relevant Secondary Features](https://arxiv.org/abs/2208.03581) - MICCAI Cancer Prevention through early detection, Conference
2. [Clinical Segmentation for Improved Pancreatic Ductal Adenocarcinoma Detection and Segmentation](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12465/124652M/Clinical-segmentation-for-improved-pancreatic-ductal-adenocarcinoma-detection-and-segmentation/10.1117/12.2654164.short) - SPIE Medical Imaging, Conference
3. Improved Pancreatic Cancer Detection and Localization on CT scans: A Computer-Aided Detection model utilizing Secondary Features - TBD, Journal


# pytorch-3dunet

PyTorch implementation of 3D U-Net and its variants:

- `UNet3D` Standard 3D U-Net based on [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650)

- `ResidualUNet3D` Residual 3D U-Net based on [Superhuman Accuracy on the SNEMI3D Connectomics Challenge](https://arxiv.org/pdf/1706.00120.pdf)

- `ResidualUNetSE3D` Similar to `ResidualUNet3D` with the addition of Squeeze and Excitation blocks based on [Deep Learning Semantic Segmentation for High-Resolution Medical Volumes](https://ieeexplore.ieee.org/abstract/document/9425041). Original squeeze and excite paper: [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)

The code allows for training the U-Net for both: **semantic segmentation** (binary and multi-class) and **regression** problems (e.g. de-noising, learning deconvolutions).

## 2D U-Net
2D U-Net is also supported, see [2DUnet_confocal](resources/2DUnet_confocal_boundary) or [2DUnet_dsb2018](resources/2DUnet_dsb2018/train_config.yml) for example configuration. 
Just make sure to keep the singleton z-dimension in your H5 dataset (i.e. `(1, Y, X)` instead of `(Y, X)`) , because data loading / data augmentation requires tensors of rank 3.
The 2D U-Net itself uses the standard 2D convolutional layers instead of 3D convolutions with kernel size `(1, 3, 3)` for performance reasons.

## Input Data Format
The input data should be stored in HDF5 files. The HDF5 files for training should contain two datasets: `raw` and `label`. Optionally, when training with `PixelWiseCrossEntropyLoss` one should provide `weight` dataset.
The `raw` dataset should contain the input data, while the `label` dataset the ground truth labels. The optional `weight` dataset should contain the values for weighting the loss function in different regions of the input and should be of the same size as `label` dataset.
The format of the `raw`/`label` datasets depends on whether the problem is 2D or 3D and whether the data is single-channel or multi-channel, see the table below:

|                | 2D           | 3D           |
|----------------|--------------|--------------|
| single-channel | (1, Y, X)    | (Z, Y, X)    |
| multi-channel  | (C, 1, Y, X) | (C, Z, Y, X) |


## Prerequisites
- NVIDIA GPU
- CUDA CuDNN

### Running on Windows/OSX
`pytorch-3dunet` is a cross-platform package and runs on Windows and OS X as well.


## Installation
- The easiest way to install `pytorch-3dunet` package is via conda/mamba:
```
conda install -c conda-forge mamba
mamba create -n pytorch-3dunet -c pytorch -c nvidia -c conda-forge pytorch pytorch-cuda=12.1 pytorch-3dunet
conda activate pytorch-3dunet
```
After installation the following commands are accessible within the conda environment:
`train3dunet` for training the network and `predict3dunet` for prediction (see below).

- One can also install directly from source:
```
python setup.py install
```

### Installation tips
Make sure that the installed `pytorch` is compatible with your CUDA version, otherwise the training/prediction will fail to run on GPU. 

## Train
Given that `pytorch-3dunet` package was installed via conda as described above, one can train the network by simply invoking:
```
train3dunet --config <CONFIG>
```
where `CONFIG` is the path to a YAML configuration file, which specifies all aspects of the training procedure. 

In order to train on your own data just provide the paths to your HDF5 training and validation datasets in the config.

* sample config for 3D semantic segmentation (cell boundary segmentation): [train_config_segmentation.yaml](resources/3DUnet_confocal_boundary/train_config.yml)
* sample config for 3D regression task (denoising): [train_config_regression.yaml](resources/3DUnet_denoising/train_config_regression.yaml)
* more configs can be found in [resources](resources) directory

One can monitor the training progress with Tensorboard `tensorboard --logdir <checkpoint_dir>/logs/` (you need `tensorflow` installed in your conda env), where `checkpoint_dir` is the path to the checkpoint directory specified in the config.

### Training tips
1. When training with binary-based losses, i.e.: `BCEWithLogitsLoss`, `DiceLoss`, `BCEDiceLoss`, `GeneralizedDiceLoss`:
The target data has to be 4D (one target binary mask per channel).
When training with `WeightedCrossEntropyLoss`, `CrossEntropyLoss`, `PixelWiseCrossEntropyLoss` the target dataset has to be 3D, see also pytorch documentation for CE loss: https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
2. `final_sigmoid` in the `model` config section applies only to the inference time (validation, test):
   * When training with `BCEWithLogitsLoss`, `DiceLoss`, `BCEDiceLoss`, `GeneralizedDiceLoss` set `final_sigmoid=True`
   * When training with cross entropy based losses (`WeightedCrossEntropyLoss`, `CrossEntropyLoss`, `PixelWiseCrossEntropyLoss`) set `final_sigmoid=False` so that `Softmax` normalization is applied to the output.

## Prediction
Given that `pytorch-3dunet` package was installed via conda as described above, one can run the prediction via:
```
predict3dunet --config <CONFIG>
```

In order to predict on your own data, just provide the path to your model as well as paths to HDF5 test files (see example [test_config_segmentation.yaml](resources/3DUnet_confocal_boundary/test_config.yml)).

### Prediction tips
1. If you're running prediction for a large dataset, consider using `LazyHDF5Dataset` and `LazyPredictor` in the config. This will save memory by loading data on the fly at the cost of slower prediction time. See [test_config_lazy](resources/3DUnet_confocal_boundary/test_config_lazy.yml) for an example config.
2. If your model predicts multiple classes (see e.g. [train_config_multiclass](resources/3DUnet_multiclass/train_config.yaml)), consider saving only the final segmentation instead of the probability maps which can be time and space consuming.
   To do so, set `save_segmentation: true` in the `predictor` section of the config (see [test_config_multiclass](resources/3DUnet_multiclass/test_config.yaml)).

## Data Parallelism
By default, if multiple GPUs are available training/prediction will be run on all the GPUs using [DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html).
If training/prediction on all available GPUs is not desirable, restrict the number of GPUs using `CUDA_VISIBLE_DEVICES`, e.g.
```bash
CUDA_VISIBLE_DEVICES=0,1 train3dunet --config <CONFIG>
``` 
or
```bash
CUDA_VISIBLE_DEVICES=0,1 predict3dunet --config <CONFIG>
```

## Supported Loss Functions

### Semantic Segmentation
- `BCEWithLogitsLoss` (binary cross-entropy)
- `DiceLoss` (standard `DiceLoss` defined as `1 - DiceCoefficient` used for binary semantic segmentation; when more than 2 classes are present in the ground truth, it computes the `DiceLoss` per channel and averages the values)
- `BCEDiceLoss` (Linear combination of BCE and Dice losses, i.e. `alpha * BCE + beta * Dice`, `alpha, beta` can be specified in the `loss` section of the config)
- `CrossEntropyLoss` (one can specify class weights via the `weight: [w_1, ..., w_k]` in the `loss` section of the config)
- `PixelWiseCrossEntropyLoss` (one can specify per-pixel weights in order to give more gradient to the important/under-represented regions in the ground truth; `weight` dataset has to be provided in the H5 files for training and validation; see sample config in [train_config.yml](resources/3DUnet_confocal_boundary_weighted/train_config.yml)
- `WeightedCrossEntropyLoss` (see 'Weighted cross-entropy (WCE)' in the below paper for a detailed explanation)
- `GeneralizedDiceLoss` (see 'Generalized Dice Loss (GDL)' in the below paper for a detailed explanation) Note: use this loss function only if the labels in the training dataset are very imbalanced e.g. one class having at least 3 orders of magnitude more voxels than the others. Otherwise, use standard `DiceLoss`.

For a detailed explanation of some of the supported loss functions see:
[Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/pdf/1707.03237.pdf).

### Regression
- `MSELoss` (mean squared error loss)
- `L1Loss` (mean absolute error loss)
- `SmoothL1Loss` (less sensitive to outliers than MSELoss)
- `WeightedSmoothL1Loss` (extension of the `SmoothL1Loss` which allows to weight the voxel values above/below a given threshold differently)


## Supported Evaluation Metrics

### Semantic Segmentation
- `MeanIoU` (mean intersection over union)
- `DiceCoefficient` (computes per channel Dice Coefficient and returns the average)
If a 3D U-Net was trained to predict cell boundaries, one can use the following semantic instance segmentation metrics
(the metrics below are computed by running connected components on threshold boundary map and comparing the resulted instances to the ground truth instance segmentation): 
- `BoundaryAveragePrecision` (Average Precision applied to the boundary probability maps: thresholds the output from the network, runs connected components to get the segmentation and computes AP between the resulting segmentation and the ground truth)
- `AdaptedRandError` (see http://brainiac2.mit.edu/SNEMI3D/evaluation for a detailed explanation)
- `AveragePrecision` (see https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric)

If not specified `MeanIoU` will be used by default.


### Regression
- `PSNR` (peak signal to noise ratio)
- `MSE` (mean squared error)

## Examples



# Datasets
We use the [Medical Decathlon](http://medicaldecathlon.com/) dataset - Task 07 Pancreas & Tumour. A few cases were supplimented with additional annotations of the  pancreatic duct, common bile duct, pancreas and pancreatic tumour for this work. The new annotations and corresponding CT volumes (nifti) can be downloaded [here](https://drive.google.com/drive/folders/1dVXYN7i3b0nNEvFDnEYtScbZzKhezxx1?usp=sharing).

# Code
The approach is based on [https://github.com/wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet).
For any further details feel free to reach out.

