# Dice Finding

Object detection and pose finding for [dice](https://en.wikipedia.org/wiki/Dice), supported by synthetic datasets of 3D rendered images.

This project was designed for learning object detection via [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network), performing full 3D pose reconstruction with help from traditional computer vision methods. By using 3D rendered images of dice, we can easily create large, annotated datasets, at the cost of image variety.

We use pyrender to generate annotated image datasets, Tensorflow and a modified version of the [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md) to find bounding boxes and associated classes, and OpenCV to perform precise pose detection given a camera intrinsic matrix.

## Description

We consider the case of dice lying flat on a horizontal surface, such as a table, captured from above by a camera without any roll around its forward axis (upwards in the world projects to upwards in the image). While slightly restrictive, this covers most of the configurations in which dice will be viewed, and allows us to make important simplifying assumptions.

With this setup, the camera will always be able be able to see the top face of each die (assuming all are the same size). We then aim to classify the dice in in image from 1 through 6, depending on the value of their top face. We set these as the object detection classes, along with class 0 representing the dots of the dice.

In addition to the standard annotations of bounding box and class label for each dot and die, we also add the rotation of each die around the vertical (y) axis relative to the direction to the camera. We also encode the camera intrinsic matrix and distortion coefficients for the camera, to allow for calibrated camera pose finding.

By determining the class of die (thereby its upwards pointing face) and the y rotation relative to the camera, we can reconstruct part of the orientation of each die relative to the camera.

By combining this approximate orientation with the known positions of the dots and their detected image positions, we try to match known and found dots and reconstruct the full 3D pose of the die.

We choose the [CenterNet](https://arxiv.org/abs/1904.07850) architecture due to its simple and anchor-free nature. Since this is designed as a learning project, it was important to choose an architecture that easily allowed adding additional prediction heads as needed to support pose reconstruction.

## Instructions

### Installation

Download or clone this repository.

[Install Tensorflow](https://www.tensorflow.org/install/pip) (tested using Version 2.2) to the virtual environment to be used.

Install the other packages in the [requirements](requirements.txt) to the virtual environment to be used (tested using given versions):

|package |numpy |matplotlib |pycpd |pyrender |trimesh |opencv_python_headless |
|--|--|--|--|--|--|--|
|version |1.18.1 |3.1.3 |2.0.0 |0.1.43 |3.7.7 |1.4.1 |4.3.0.36 |

Note: this repository contains a modified version of the Tensorflow Object Detection API, so you should not need to [install](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md#installation) it. Note that the [protocol buffers](https://github.com/tensorflow/models/tree/master/research/object_detection/protos) have been compiled for Python.

### Generating TFRecord Dataset

Run `dice_finding/DiceLabelling.py` to generate images and annotations to a TFRecord file. Update the num_examples and current_offset as desired. Each die uses the current_offset to determine position and rotation values from a [Halton sequence](https://en.wikipedia.org/wiki/Halton_sequence), for determistic yet low-discrepancy results.

Generate one to use for training, and another for evaluation, with a larger current_offset set so that the datasets don't overlap.

### Training and Evaluation

Ensure `tf_config/centernet_pipeline.config` has input_path set for train_input_reader and eval_input_reader, based on the datasets generated as described above.

Training and evaluation is done using a modified version of the CenterNet object detection architecture from the [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md). Follow the [instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md) to run training and evaluation. Update the `centernet_pipeline.config` as desired.

To run through the Visual Studio Python Project, ensure that the command-line argument --model_dir is set in project settings, with --checkpoint_dir also provided to run evaluation.

### Inference and Visualization

Run `dice_finding/InferenceVisualization.py` to perform inference, 3D pose finding, and visualization on the evaluation dataset specified in the `centernet_pipeline` configuration. It will also write VGG Image Annotator-format .csv annotations with the object detection bounding box (and point) results.

## Future Work

* Extend unit tests for CenterNet and evaluation to consider y rotation output.
* Unit tests for dice_finding modules.
* Annotate full 3D pose into TFRecords.
* Relax assumption of vertically-oriented camera in pose finding and image datasets.
* Add scenery, lighting, and other variability to rendered images.
* Add other methods of data augmentation for use in training.
* Perform training, evaluation, and inference using real photos of dice.
* Add a mode to handle video, where pose finding uses results from previous frames.

## Packages Used

* [numpy](https://pypi.org/project/numpy/) for fundamental math and linear algebra
* [matplotlib](https://pypi.org/project/matplotlib/) for plotting
* [pycpd](https://pypi.org/project/pycpd/) for Coherent Point Drift registration
* [pyrender](https://pypi.org/project/pyrender/) for generating rendered images
* [trimesh](https://pypi.org/project/trimesh/) for loading triangular meshes
* [SciPy](https://pypi.org/project/scipy/) for minimization and linear sum assignment
* [OpenCV](https://pypi.org/project/opencv-python/) for computer vision (point projection, PnP)
* [Tensorflow](https://pypi.org/project/tensorflow/) for Convolutional Neural Network architecture, training, and inference

## Authors

### Dice Finding

Tyler Dodds ([@GitHub TylerDodds](https://github.com/TylerDodds))

### Third Party

The [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md) is reproduced here in `object_detection`; it is modified by the Dice Finding authors to enable prediction of dice y rotation values.

An implementation of a Halton Sequence by [Pamphile Tupui ROY](https://gist.github.com/tupui/cea0a91cc127ea3890ac0f002f887bae) is used in `dice_finding/MathUtil.py`, as is an implementation of orthogonalization by [Anmol Kabra](https://gist.github.com/tupui/cea0a91cc127ea3890ac0f002f887bae) in `dice_finding/GramSchmidt.py`.

## License

[Apache License 2.0](LICENSE)