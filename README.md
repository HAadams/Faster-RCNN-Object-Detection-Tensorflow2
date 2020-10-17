# Faster-RCNN 2D Object Detection with Tensorflow v2
Instructions for training a Tensorflow v2-compatible model from the [Tensorflow v2 Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). We use the ***Faster R-CNN ResNet50 V1 640x640*** model for this tutorial along with Berkely's DeepDrive Images and Labels (2020 version). This page includes higih level instructions to:

    1. Download images and labels
    2. Install Object Detection API
    3. Create Label map 
    4. Create TFRecords
    5. Pipeline config modification
    6. Train the model
    7. Evaluate the model
    8. Export the model for later use
    9. Model inference using the exported model

    NOTE: 
    It is recommended to use Google Colab. 
    Most of the base requirements are already installed in the hosted environment.


## Install Object Detection Module
1. Install Tensroflow Object Detection module.
   
Run the following commands (remove %%bash if not running in Google Colab)
```bash
    %%bash
    git clone --depth 1 https://github.com/tensorflow/models
    cd models/research/
    protoc object_detection/protos/*.proto --python_out=.
    cp object_detection/packages/tf2/setup.py .
    python -m pip install .

```
## Download Dataset
    NOTE: After downloading the dataset, all images were moved to a single folder.

### Labels
1. Go to [Berekly DeepDrive](https://bdd-data.berkeley.edu) and click Download and make an account
2. Download "Detection 2020 Labels"
### Images
#### Option 1
1. solesensei's bdd100k Images from [Kaggle](https://www.kaggle.com/solesensei/solesensei_bdd100k).
#### Option 2
1. Images tab at [Berekly DeepDrive](https://bdd-data.berkeley.edu)

## Download Object Detection Model
1. [Tensorflow v2 Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
2. Unzip the file

I used faster_rcnn_resnet50_v1_640x640_coco17_tpu

```bash
# Google Colab code
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
!tar -xf faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz

```

## Create Label Map
The labels file contains a list of items. Each item describes one label. 

    NOTE: This label map file must remain the same if model is being trained multiple times.

Example:

labels_map.pbtxt

    item {
        id: 1
        name: "rider"
    }
    item {
        id: 2
        name: "bicycle"
    }

The following code reads the label json file and extracts the labels and automatically generates a label map.

```python

import os
import sys
import json

# Open labels file to extract class names and build labels map (.pbtxt) file
f = open('bdd100k/labels/detection20/det_v2_val_release.json')
s = json.load(f)

# template for each 'item' in the pbtxt file
item = """item {{
  id: {id}
  name: "{name}"
}}
"""

# Function to build a label's class name
def get_class_name(single_label: dict):
  cls_label = single_label['category'].replace(' ', '_')
  if single_label['category'] == 'traffic light' and single_label['attributes']['trafficLightColor'][0] > 0:
      cls_label += '_'+single_label['attributes']['trafficLightColor'][1]
  
  return cls_label.lower()

# Extract all class names from the labels file and save in a set to ensure unique values
classes = set()
for image in s:
  if image['labels']:
    for label in image['labels']:
      classes.add(get_class_name(label))

# Sort to make sure each label maintains its ID across multiple runs 
classes = sorted(classes)

# Dictionary to map IDs to class names (used later to map text to id to generate tfrecrod)
class_text_to_int = dict()
# Write the .pbtxt file
labels_pbtxt_file = "label_map.pbtxt"
with open(labels_pbtxt_file, "w") as labels_pbtxt:
 for i in range(len(classes)):
    fitem = item.format(name=classes[i], id=i+1)
    class_text_to_int[classes[i]] = i+1
    labels_pbtxt.write(fitem)

print("Created labels map 'label_map.pbtxt' successfully...")
```

Output file (label_map.pbtxt)

```java
item {
  id: 1
  name: "bicycle"
}
item {
  id: 2
  name: "bus"
}
item {
  id: 3
  name: "car"
}
item {
  id: 4
  name: "motorcycle"
}
item {
  id: 5
  name: "other_person"
}
item {
  id: 6
  name: "other_vehicle"
}
item {
  id: 7
  name: "pedestrian"
}
item {
  id: 8
  name: "rider"
}
item {
  id: 9
  name: "traffic_light"
}
item {
  id: 10
  name: "traffic_light_g"
}
item {
  id: 11
  name: "traffic_light_r"
}
item {
  id: 12
  name: "traffic_light_y"
}
item {
  id: 13
  name: "traffic_sign"
}
item {
  id: 14
  name: "trailer"
}
item {
  id: 15
  name: "train"
}
item {
  id: 16
  name: "truck"
}
```

## Create TF Records
TF Records are used to store serialized binary records. In this case, each record contains information such as
1. Image binary data
2. List of x,y coordinantes of each labeled object in the image.
3. List of labels of each labeled object in the image (id and name)
4. Image file name
5. Image extension
6. Image height
7. Image width

The following code is based off of [generate_tfrecord.py](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py) and is specialized to parse Berkely DeepDrive Detection 2020 data

```python

import os
import io
import sys
import json
import numpy as np
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util

def create_tf_record(group, path):
    encoded_jpg = open(os.path.join(path, group['name']), 'rb').read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = group['name'].encode('utf8')
    image_format = b'jpg'
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []
    
    # Dont add labels to this tfrecord if it doesnt exist in the label map
    if group['labels']:
      # normalize coordinates for image resizing
      for label in group['labels']:
          x1 = label['box2d']['x1']
          x2 = label['box2d']['x2']
          y1 = label['box2d']['y1']
          y2 = label['box2d']['y2']
          # Skip bad labels 
          # if x1 >= x2 or y1 >= y2:
            # print("Malformed coordinates", x1, x2, y1, y2)
            # continue
          xmins.append(np.true_divide(x1,width))
          xmaxs.append(np.true_divide(x2,width))
          ymins.append(np.true_divide(y1,height))
          ymaxs.append(np.true_divide(y2,height))
          classes_text.append(get_class_name(label).encode('utf8'))
          classes.append(class_text_to_int[get_class_name(label)])

    # Create a tf record for this image and all the bounded boxes in it
    tf_record = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_record

def create_records(labels_path, images_path, output_record_name, limit):
    with tf.io.TFRecordWriter(output_record_name) as writer:
      # Open the label data to see what we're working with
      with open(labels_path, 'r') as labels_file:
        labels_json = json.load(labels_file)

        # load all images from labels to this TFrecord if no limit specified
        if limit is None:
          limit = len(labels_json)

        for i in range(limit):
            sys.stdout.write(f"\r{output_record_name} processed {i+1}/{len(labels_json)} images")
            tf_record = create_tf_record(labels_json[i], images_path)
            if tf_record is not None:
              writer.write(tf_record.SerializeToString())
              i += 1
    print('\nSuccessfully created the TFRecords: {}'.format(output_record_name))

train_size = None # int num of images (None: Convert all images)
train_labels_path = 'bdd100k/labels/detection20/det_v2_train_release.json'
train_image_path = 'bdd100k/images'
train_record_path = 'train.record'

test_size = None # int num of images (None: Convert all images)
test_labels_path = 'bdd100k/labels/detection20/det_v2_val_release.json'
test_image_path = 'bdd100k/images'
test_record_path = 'test.record'

for labels_path, images_path, record_name, limit in [(test_labels_path, test_image_path, test_record_path, test_size), (train_labels_path, train_image_path, train_record_path, train_size)]:
  create_records(labels_path=labels_path, images_path=images_path, output_record_name=record_name, limit=limit)

```

## Pipeline Config File

Using faster_rcnn_resnet50_v1_640x640_coco17_tpu as an example,

1. Copy the pipeline.config found in the model folder (i.e. faster_rcnn_resnet50_v1_640x640_coco17_tpu/pipeline.config) into your working directory.
2. Update the following parameters inside the pipeline config file:
   1. num_classes: 16
   2. under train_input_reader, update: 
      1. label_map_path: "path to label_map.pbtxt"
      2. input_path: "path to train.record"
   3. under eval_input_reader, update: 
      1. label_map_path: "path to label_map.pbtxt"
      2. input_path: "path to test.record"

   4. Under train_config update:
      1. batch_size: 4 
      2. num_steps: 30000 (NOTE: higher number of steps, the longer training takes)
      3. fine_tune_checkpoint: "path to faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/checkpoint/ckpt-0"
      4. fine_tune_checkpoint_type: "detection"
   5. Under eval_config
      1. batch_size: 1



## Start Training
```bash
PIPELINE_CONFIG_PATH = 'path to pipeline config file'
MODEL_OUTPUT_PATH = './output_model' # Tensorflow will create this output folder
!python models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={PIPELINE_CONFIG_PATH} \
    --model_dir={MODEL_OUTPUT_PATH} \
    --alsologtostderr

```
If you see the following logs after a few minutes, it means training is working:

```python
I1009 05:49:54.640861 139781364750208 model_lib_v2.py:652] Step 100 per-step time 0.260s loss=1.553
INFO:tensorflow:Step 200 per-step time 0.244s loss=1.249
I1009 05:50:19.954221 139781364750208 model_lib_v2.py:652] Step 200 per-step time 0.244s loss=1.249
INFO:tensorflow:Step 300 per-step time 0.245s loss=1.501
I1009 05:50:45.443155 139781364750208 model_lib_v2.py:652] Step 300 per-step time 0.245s loss=1.501
INFO:tensorflow:Step 400 per-step time 0.248s loss=1.894
I1009 05:51:11.166207 139781364750208 model_lib_v2.py:652] Step 400 per-step time 0.248s loss=1.894
INFO:tensorflow:Step 500 per-step time 0.270s loss=1.251
I1009 05:51:36.635663 139781364750208 model_lib_v2.py:652] Step 500 per-step time 0.270s loss=1.251
INFO:tensorflow:Step 600 per-step time 0.241s loss=1.397

```

    Tensorflow will save progress in the output folder continously so that if training is interrupted or ends, you can start from where you left off by running the same command.


## Evaluation
Specifiying the --checkpoint_dir argument tells Tensorflow to run evaluation instead of training.

```bash
!python models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={PIPELINE_CONFIG_PATH} \
    --model_dir={MODEL_OUTPUT_PATH} \
    --checkpoint_dir={MODEL_OUTPUT_PATH} \
    --alsologtostderr
```

Example Evaluation results:

```python
...
...
...
I1009 17:40:06.071091 140592774539136 model_lib_v2.py:799] Finished eval step 9600
INFO:tensorflow:Finished eval step 9700
I1009 17:40:13.232145 140592774539136 model_lib_v2.py:799] Finished eval step 9700
INFO:tensorflow:Finished eval step 9800
I1009 17:40:20.368026 140592774539136 model_lib_v2.py:799] Finished eval step 9800
INFO:tensorflow:Finished eval step 9900
I1009 17:40:27.592002 140592774539136 model_lib_v2.py:799] Finished eval step 9900
INFO:tensorflow:Performing evaluation on 10000 images.
I1009 17:40:34.615332 140592774539136 coco_evaluation.py:282] Performing evaluation on 10000 images.
creating index...
index created!
INFO:tensorflow:Loading and preparing annotation results...
I1009 17:40:34.780122 140592774539136 coco_tools.py:116] Loading and preparing annotation results...
INFO:tensorflow:DONE (t=1.70s)
I1009 17:40:36.485057 140592774539136 coco_tools.py:138] DONE (t=1.70s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=384.82s).
Accumulating evaluation results...
DONE (t=50.25s).


Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.142
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.288
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.120
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.219
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.396
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.164
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.284
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.297
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.113
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.385
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.570
INFO:tensorflow:Eval metrics at step 150000
```

## Inference
### Export the model for inference

```python
EXPORTED_MODEL_PATH = './exported_model' # Path to save the exported model
!python models/research/object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path {PIPELINE_CONFIG_PATH} --trained_checkpoint_dir {MODEL_OUTPUT_PATH}  --output_directory {EXPORTED_MODEL_PATH}

```
### Load the model
```python
import tensorflow as tf
import os
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load(os.path.join(EXPORTED_MODEL_PATH, 'saved_model'))

from object_detection.utils import label_map_util
label_map_path='path to the label map pbtxt file'
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
```

### Detect Objects

```python
import time
import pathlib
import requests
from io import BytesIO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from object_detection.utils import visualization_utils as viz_utils

%matplotlib inline

resp = requests.get('https://images.techhive.com/images/article/2015/09/garmin_nuvicam_dashcam_day-100615653-orig.png')

image = Image.open(BytesIO(resp.content))
(im_width, im_height) = image.size
image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)

plt.rcParams['figure.figsize'] = [42, 21]
label_id_offset = 1
image_np_with_detections = image_np.copy()
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'][0].numpy(),
      detections['detection_classes'][0].numpy().astype(np.int32),
      detections['detection_scores'][0].numpy(),
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.40,
      agnostic_mode=False)

plt.figure(figsize=(18,24))
plt.imshow(image_np_with_detections)
plt.show()

```

![alt text](https://github.com/HAadams/Faster-RCNN-Object-Detection-Tensorflow2/blob/main/detection1.png?raw=true)
![alt text](https://github.com/HAadams/Faster-RCNN-Object-Detection-Tensorflow2/blob/main/detection2.png?raw=true)
