#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/meituan/YOLOv6/blob/main/turtorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 
# This is the official YOLOv6 notebook by MeiTuan, and is freely available for redistribution under the [GPL-3.0 license](https://choosealicense.com/licenses/gpl-3.0/). 
# For more information please visit https://github.com/meituan/YOLOv6. Thank you!

# # Introduction
# 
# YOLOv6 is a single-stage object detection framework dedicated to industrial applications, with hardware-friendly efficient design and high performance.
# 
# YOLOv6 is composed of the following methods:
# 
# Hardware-friendly Design for Backbone and Neck
# Efficient Decoupled Head with SIoU Loss

# # Setup
# Clone repo and install dependencies.

# In[ ]:


get_ipython().system('git clone https://github.com/meituan/YOLOv6.git')
get_ipython().run_line_magic('cd', 'YOLOv6')
get_ipython().run_line_magic('pip', 'install -r requirements.txt')


# # Inference
# First, download a pretrained model from the YOLOv6 [release](https://github.com/meituan/YOLOv6/releases).
# 
# 

# In[ ]:


# Download a pretrained model
import torch
torch.hub.download_url_to_file('https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6s.pt', 'yolov6s.pt')


# Second, run inference with `tools/infer.py`, and saving results to `runs/inference`. Example inference sources are:
# 
# ```shell
# python tools/infer.py --weights yolov6s.pt --source img.jpg / imgdir
#                                 yolov6n.pt
# ```

# In[ ]:


get_ipython().system('python tools/infer.py --weights yolov6s.pt --source data/images/image1.jpg')
# show image
from google.colab.patches import cv2_imshow, cv2
img = cv2.imread('runs/inference/exp/image1.jpg')
cv2_imshow(img)


# ### You can also use torch.hub style to load the pretrained model or custom model to inference.

# In[4]:


import torch 

model_local = torch.hub.load('.', 'yolov6n', source='local')  


# In[5]:


img_path = 'data/images/image1.jpg'


# In[6]:


prediction = model_local.predict(img_path)
#prediction = model_custom.predict(img_path)
display(prediction)


# In[7]:


model_local.show_predict(img_path)


# # Validate
# Validate a model's accuracy on [COCO](https://cocodataset.org/#home) val or test-dev datasets. Models are downloaded automatically from the [latest YOLOv6 release](https://github.com/meituan/YOLOv6/releases). 

# ## COCO val
# Download COCO val 2017 dataset (1GB - 5000 images), and test model accuracy.

# In[ ]:


# Download COCO val
import torch
torch.hub.download_url_to_file('https://ultralytics.com/assets/coco2017val.zip', 'tmp.zip')
get_ipython().system('unzip -q tmp.zip -d ../ && rm tmp.zip')


# In[ ]:


# Run yolov6x on coco val
get_ipython().system('python tools/eval.py --weights yolov6s.pt --data data/coco.yaml --img 640')


# # Train coco data
# conf: select config file to specify network/optimizer/hyperparameters
# 
# data: prepare [COCO](http://cocodataset.org)  dataset, [YOLO format coco labes](https://github.com/meituan/YOLOv6/releases/download/0.1.0/coco2017labels.zip) and specify dataset paths in data.yaml
# 
# make sure your dataset structure as fellows:
# ```shell
# ├── coco
# │   ├── annotations
# │   │   ├── instances_train2017.json
# │   │   └── instances_val2017.json
# │   ├── images
# │   │   ├── train2017
# │   │   └── val2017
# │   ├── labels
# │   │   ├── train2017
# │   │   ├── val2017
# │   ├── LICENSE
# │   ├── README.txt
# ```

# ## COCO datasets

# In[ ]:


# Download coco datasets and need about 30mins.
get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('cd', 'coco/images')
get_ipython().system('wget http://images.cocodataset.org/zips/train2017.zip')
get_ipython().system('wget http://images.cocodataset.org/zips/val2017.zip')
get_ipython().system('wget http://images.cocodataset.org/zips/test2017.zip')
get_ipython().system('unzip train2017.zip && rm train2017.zip')
get_ipython().system('unzip val2017.zip && rm val2017.zip')
get_ipython().system('unzip test2017.zip && rm test2017.zip')


# In[ ]:


# Before running, you need to make sure you're in the YOLOv6 root directory.
get_ipython().run_line_magic('cd', '../../YOLOv6')
# Train YOLOv6s on COCO for 30 epochs
get_ipython().system('python tools/train.py --img 640 --batch 32 --epochs 30 --conf configs/yolov6s.py --data data/coco.yaml')


# ## COCO128 datasets
# You need create a new file `coco128.yaml` under the folder `./data`.The details are as follows:
# 
# ```
# # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
# path: ../coco128  # dataset root dir
# train: images/train2017  # train images (relative to 'path') 128 images
# val: images/train2017  # val images (relative to 'path') 128 images
# test:  # test images (optional)
# 
# # Classes
# nc: 80  # number of classes
# names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#         'hair drier', 'toothbrush']  # class names
# ```

# In[ ]:


coco128 = """# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../coco128  # dataset root dir
train: ../coco128/images/train2017  # train images 128 images
val: ../coco128/images/train2017  # val images 128 images
test:  # test images (optional)

# Classes
nc: 80  # number of classes
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names
"""

with open('data/coco128.yaml', 'w') as f:
  f.write(coco128)


# In[ ]:


# Download coco128 datasets
torch.hub.download_url_to_file('https://ultralytics.com/assets/coco128.zip', 'tmp.zip')
get_ipython().system('unzip -q tmp.zip -d ../ && rm tmp.zip')

# torch.hub.download_url_to_file('https://drive.google.com/file/d/1HICm-rrsdp89GNpFbzcwksHRtDx10McK/view?usp=sharing', 'tmp.zip')
# !unzip -q tmp.zip -d ../ && rm tmp.zip


# In[ ]:


# Train YOLOv6s on COCO128 for 100 epochs
get_ipython().system('python tools/train.py --img 640 --batch 32 --epochs 100 --data data/coco128.yaml')


# In[ ]:


# Tensorboard  (optional)
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir runs/train')


# # Train Custom Data
# This guidence explains how to train your own custom data with YOLOv6 (take fine-tuning YOLOv6-s model for example).

# ## Prepare your own dataset

# **Step 1** Prepare your own dataset with images. For labeling images, you can use tools like [Labelme](https://github.com/wkentaro/labelme).
# 
# **Step 2** Generate label files in YOLO format.
# 
# One image corresponds to one label file, and the label format example is presented as below.
# 
# ```json
# # class_id center_x center_y bbox_width bbox_height
# 0 0.300926 0.617063 0.601852 0.765873
# 1 0.575 0.319531 0.4 0.551562
# ```
# 
# - Each row represents one object.
# - Class id starts from `0`.
# - Boundingbox coordinates must be in normalized `xywh` format (from 0 - 1). If your boxes are in pixels, divide `center_x` and `bbox_width` by image width, and `center_y` and `bbox_height` by image height.
# 
# **Step 3** Organize directories.
# 
# Organize your directory of custom dataset as follows:
# 
# ```shell
# custom_dataset
# ├── images
# │   ├── train
# │   │   ├── train0.jpg
# │   │   └── train1.jpg
# │   ├── val
# │   │   ├── val0.jpg
# │   │   └── val1.jpg
# │   └── test
# │       ├── test0.jpg
# │       └── test1.jpg
# └── labels
#     ├── train
#     │   ├── train0.txt
#     │   └── train1.txt
#     ├── val
#     │   ├── val0.txt
#     │   └── val1.txt
#     └── test
#         ├── test0.txt
#         └── test1.txt
# ```
# 
# **Step 4** Create `dataset.yaml` in `$YOLOv6_DIR/data`.
# 
# ```yaml
# # Please insure that your custom_dataset are put in same parent dir with YOLOv6_DIR
# train: ../custom_dataset/images/train # train images
# val: ../custom_dataset/images/val # val images
# test: ../custom_dataset/images/test # test images (optional)
# 
# # whether it is coco dataset, only coco dataset should be set to True.
# is_coco: False
# 
# # Classes
# nc: 20  # number of classes
# names: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
#         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']  # class names
# ```

# ## Create a config file

# 
# We use a config file to specify the network structure and training setting, including  optimizer and data augmentation hyperparameters.
# 
# If you create a new config file, please put it under the configs directory.
# Or just use the provided config file in `$YOLOV6_HOME/configs/*_finetune.py`.
# 
# ```python
# ## YOLOv6s Model config file
# model = dict(
#     type='YOLOv6s',
#     pretrained='./weights/yolov6s.pt', # download pretrain model from YOLOv6 github if use pretrained model
#     depth_multiple = 0.33,
#     width_multiple = 0.50,
#     ...
# )
# solver=dict(
#     optim='SGD',
#     lr_scheduler='Cosine',
#     ...
# )
# 
# data_aug = dict(
#     hsv_h=0.015,
#     hsv_s=0.7,
#     hsv_v=0.4,
#     ...
# )
# ```
# 
# 

# ## Train

# In[ ]:


get_ipython().system('python tools/train.py --batch 256 --conf configs/yolov6s_finetune.py --data data/data.yaml')


# # Test Speed

# In[ ]:


get_ipython().system('python tools/eval.py --data data/coco128.yaml --batch 32 --weights yolov6s.pt --task speed')

