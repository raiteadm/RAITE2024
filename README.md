Blue-Purdue
# RAITE 2024 Object Detectors
This repository contains the models used by Purdue's research group for the RAITE 2024 testing. It contains code and weights for 2 models/algorithms:

* An RGB object detector 
* An IR object detector 

Both are from the **EfficientDet** family, specifically the D3 variant. EfficientDet-D3 was chosen for its mAP performance on the MSCOCO dataset, while still retaining real-time latency for image processing.

The repository contains code that allows you to run the models on images, videos, or both. Please see the [Usage](#usage) section to see how to run on a set of data.

The IR model's weights can be found in `./weights/efficientdet-d3_flir.pth`. The weights for this model have been fine-tuned on the [FLIR](https://www.flir.com/oem/adas/adas-dataset-form/?srsltid=AfmBOoo5d8LHBfpg4zmbkF99bG93UtiZe9VPdRWRoha3uxQCKOQODHEa) dataset. The RGB model weights can be found in `./weights/efficientdet-d3.pth`.

The complete model can be found in the `EfficientDetBackbone` class in `backbone.py`. This may be a good starting point for understanding model architecture, beyond reading the EfficientDet paper. To edit model weights at runtime, please see lines *54-56* of `main.py`, where the model weights are loaded before inference time.

For training a model, please refer back to the original repository that this repository was adapted from, [Yet Another EfficientDet Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch).

## Acknowledgements
This repository was adapted from [Yet Another EfficientDet Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch), a PyTorch implementation of EfficientDet. EfficientDet is a model architecture designed by developers at Google, which can be found at [this](https://github.com/google/automl/tree/master/efficientdet) repository. Their paper introducing the model architecture can be found [here](https://arxiv.org/abs/1911.09070).

* **Yet Another EfficientDet Pytorch:** https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
* **EfficientDet (Tensorflow):** https://github.com/google/automl/tree/master/efficientdet
* **EfficientDet (Original Paper):** https://arxiv.org/abs/1911.09070

## Setup
Below are steps for setting up the repository.

> It is possible to set it up using another package manager other than Anaconda, as well as change the python version, but do so at your own discretion.

1. Install CUDA and CUDNN libraries.
2. Install Anaconda: https://www.anaconda.com/download
3. Create an Anaconda environment (and activate it):

`> conda create -n blue_purdue python=3.11 -y`

`> conda activate blue_purdue`

4. Install PyTorch - you will need to go to https://pytorch.org/ To find which version to install for your version of CUDA and CUDNN. Below is the pytorch for *CUDA 11.8*:

`> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

5. Install the required python packages:

`> pip3 install -r ./requirements.txt`

6. Validate the installation for GPU succeeded:

`> python ./test_gpu.py`

> There is a chance that your CUDA version is incompatible with the version of PyTorch used in this repository. In this case, you will need to find the corresponding version of PyTorch for your CUDA version to run on GPU. Once you have, update the `requirements.txt` file accordingly.

> Additionally, `main.py` was written for python version 3.11. The biggest issue you will come across with older versions of python will be errors related to *Type Hints*. If this becomes an issue, you can either 1. Reformat `main.py` to use the correct python *Type Hints* for your python version, or 2. Remove all type hints that give you errors at runtime, at the cost of readability.

## Usage
Please read `main.py` to find out more about each command line argument, or run the command: `> python main.py --help` in the terminal.

This is an example command, which:
* Loads a D3 type efficientdet model
* loads pre-trained weights from the file path (`./weights/efficientdet-d3.pth`)
* loads images, videos, or both from the given data path (`./data/`)
* writes the data to a folder (`./inference/`)
* processes the images in sets of 32 (based on GPU capabilities)

`> python ./main.py --model_type=3 --weights_path="./weights/efficientdet-d3.pth" --data_path="./data/" --write_data=True --output_dir="./inference/" --batch_size=1`
