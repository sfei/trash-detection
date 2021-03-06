# Trash Detection Overview

The Trash Detection git repository contains a variety of scripts and tools for generating data sets, initiating the TensorFlow model building process, exporting model files, and running predictions on new sample data.

More information about the project can be found at [our Trash Monitoring project website](http://trashmonitoring.org/).

## Software Dependencies

Trash detection scripts were developed and run using TensorFlow 1.13.1 and the TensorFlow [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)]. Unfortunately Google has revised the Object Detection API github repository and we could not successfully find an appropriate version that pairs with TF 1.13.1. 

We've included the [Anaconda Enviornment](https://www.anaconda.com/) file TensorFlow-GPU-1.13.1.yml which lists package versions used for this work.

While it's advised you refer to the anaconda environment file for specific versions, the following libraries were necessary for installing and running TensorFlow, the Object Detection API and our scripts:
* Protobuf 3.0.0
* Python-tk
* Pillow 1.0
* lxml
* tf Slim (which is included in the "tensorflow/models/research/" checkout)
* Jupyter notebook
* Matplotlib
* Tensorflow (tested with v1.12 and v1.13)
* Cython
* contextlib2
* cocoapi

## Installation

The basic installation process follows:
1. Install TensorFlow, anaconda recommended
2. Download the Object Detection API (must work with TF v1.12 or TF v1.13)
3. Generate protobuf files
4. Download base model from the model zoo. We used the faster_rcnn_inception_resnet_v2_atrous_coco model as our base model for transfer learning.
5. Copy config-template.ini to config.ini to run scripts using config file support. Some scripts have paths entered as global variables within their respective scripts, some use config.ini.

## Running

Our model building and evaluation workflow went as follows:
1. Train using trainers/model_main.py, parameters are managed with pipeline.config. We've included a windows bash script for initiating this process, see trainers/run_training.cmd
2. Monitor model training using Tensorboard ([see Tensorboard documentation](https://www.tensorflow.org/tensorboard))
3. Use eval/export_checkpoint.py to export the model inference graph from a current checkpoint
4. Use eval/trash_classifier.py to run an exported model over novel imagery.

## Directory Structure

There are three core directories:
* eval - Python scripts for exporting and inferencing using a trained TensorFlow model.
* models - Model used for trash detection experiments.
* trainers - Python script for training trash detection models
* utils - Utility scripts for a variety of tasks including generating TensorFlow records and obtaining dataset statistics.

## Annotation

Initial annotations for the trash detection project were created using [LabelImg](https://github.com/tzutalin/labelImg). These annotations were stored in the PascalVOC format (XML files) but have since been converted into a CSV file. For other machine learning projects we've begun using [CVAT](https://github.com/openvinotoolkit/cvat) for annotation work.

## Data and Model Files
Full original dataset, includes un-cropped drone images, annotations in PascalVOC format, and orthorectified images for all associated site surveys:
[Full Original Dataset](https://filecloud.sfei.org/index.php/s/AELpbMnQfzjfTYX)

Cropped image data and annotation data (stored as CSV) can be found at:
[Annotations and Image Data](http://filecloud.sfei.org/index.php/s/DRpy3qQaZxpyMXA)

TensorFlow records created using cropped images and annotation data:
[TensorFlow Records](http://filecloud.sfei.org/index.php/s/PcBZ3DR7Pt3DjTk)

Trained trash detection model files can be found at:
[Model Files](http://filecloud.sfei.org/index.php/s/KP4yZ4Nd93bmWBf)

Trained trash detection model files in a TensorFlow serving model format can be found at:
[TF Serving Model Files](http://filecloud.sfei.org/index.php/s/myAZbArckS6CiNY)

Base model used for transfer learning:
[faster_rcnn_inception_resnet_v2_atrous_coco](http://filecloud.sfei.org/index.php/s/Fsjz7EC5yLBFbaE)
