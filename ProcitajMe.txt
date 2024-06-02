# DATASET

CDNET dataset used for training both models can be found on following google drive [link](https://drive.google.com/file/d/1RIIpdrAUXZRuoOgHMIr-BSsJxSjqiKC8/view)

# YOLOV8 MODEL

Example of final product can be found in [this](https://drive.google.com/file/d/1G_cXZrabnY8RNWDeOBumJuLoKmYFO-Eo/view?usp=sharing) video.
There are also 3 pretrained models as mentioned in the paper.

## Training

For training YOLO models requires Images and their appropriate labels marking crossroads and/or guide arrows. Labels are in format 

label x_center/width y_center/height width/image_width, height/image_height

for example:
0 0.460156 0.687500 0.833594 0.152778

Afterwards, the dataset is linked to Train.py using .yaml file formated as:

"
train: ../dataset/images/train

val: ../dataset/images/test

train_labels: ../dataset/labels/train

val_labels: ../dataset/labels/train

nc: 2

#class names

names: ['crosswalk', 'guide_arrows']"

Finaly user in Train.py decides on number of epochs and model version and training can begin.

After training summary files are created for metrics.

## Predictions

Once trained model will generate 2 .pt files "last.pt" and "best.pt". Last can be used to continue stopped training while best can be used for predictions.

Inside Predict.py user must provide path to best.pt of trained model and path to dataset to be predicted.

Dataset can include both photo directories or videos.

# FASTER R-CNN

## OBJECT DETECTION MODEL
Faster R-CNN model was used in the notebook Faster-R-CNN_crosswalk-detection.ipynb, utilizing a PyTorch pretrained implementation based on the [ Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) paper.


## DATASET PREP
The dataset contains 3,434 images, split into a 90:10 ratio with 3,080 images for training and 354 for testing. All images have corresponding labels, and there are no missing labels. In the ZebraCrossingDataset class, images were loaded with their corresponding labels.
General dataset structure:
zebra-crossing-dataset
│
├── images
│   ├── test (contains jpg images)
│   └── train (contains jpg images)
│
└── labels
    ├── test (contains txt files named the same as the corresponding images)
    └── train (contains txt files named the same as the corresponding images)

The label files contain data in the format: 'class_label x_center y_center width height'.

## TRAINING
The model was trained on three classes: 0 for background, 1 for crosswalks, and 2 for guide arrows. It was trained over 10 epochs using stochastic gradient descent with a variable learning rate that goes up to 0.005, a momentum of 0.9, and a weight decay of 0.0005.


## PREDICTIONS
The Faster R-CNN model achieved an Average Precision (AP) of 0.951 at an IoU threshold of 0.50, indicating accurate detection of objects in the crosswalk-detection dataset. In addition to predicting bounding boxes, the model also classifies objects. The final precision and recall were 0.6344 and 0.6334, respectively.
