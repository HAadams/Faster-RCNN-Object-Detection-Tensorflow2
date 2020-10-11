# 2D Object Detection
## (Traffic light, signs, cars, people, bikes...etc) Using Berkely DeepDrive Images and Labels datasets See "Images" and "Detection 2020 Labels" tabs at [Bekely DeepDrive](https://bdd-data.berkeley.edu/portal.html#download))

## All the image files and labels have been pre-downloaded and saved in my Google Drive. Because there is no direct download link from Berekly DeepDrive, saving the dataset in my Google Drive makes it much more convenient to access.


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
!cp drive/My\ Drive/bdd100k_data.tar.gz .
!tar -xf bdd100k_data.tar.gz
!rm bdd100k_data.tar.gz
```

## Custom code I wrote to build the labels map (.pbtxt) file from the validation labels file (smaller file and contains all labels)


```python
# Open annotations file in pandas
import os
import sys
import json

# Open labels file to extract class names and build labels map (.pbtxt) file
# We use the val labels file for this purpose because its a smaller file
# and it contains all the labels
f = open('bdd100k_data/labels/det_v2_val_release.json')
s = json.load(f)
# template for each 'item' in the pbtxt file
item = """item {{
  id: {id}
  name: "{name}"
}}
"""

# Function to build a label's class name
def get_class_name(single_label: dict):
  cls = single_label['category'].replace(' ', '_')
  if single_label['category'] == 'traffic light' and single_label['attributes']['trafficLightColor'][0] > 0:
      cls += '_'+single_label['attributes']['trafficLightColor'][1]
  
  return cls.lower()

# Extract all class names from the labels file and save in a set to ensure unique values
classes = set()
for image in s:
  if image['labels']:
    for label in image['labels']:
      classes.add(get_class_name(label))
classes = list(classes)

# Dictionary to map IDs to class names (used later to map text to id for tfrecrod)
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

    Created labels map 'label_map.pbtxt' successfully...



```python
!cat label_map.pbtxt
```

    item {
      id: 1
      name: "rider"
    }
    item {
      id: 2
      name: "bicycle"
    }
    item {
      id: 3
      name: "other_person"
    }
    item {
      id: 4
      name: "trailer"
    }
    item {
      id: 5
      name: "traffic_light_r"
    }
    item {
      id: 6
      name: "traffic_light_g"
    }
    item {
      id: 7
      name: "bus"
    }
    item {
      id: 8
      name: "motorcycle"
    }
    item {
      id: 9
      name: "traffic_sign"
    }
    item {
      id: 10
      name: "traffic_light"
    }
    item {
      id: 11
      name: "truck"
    }
    item {
      id: 12
      name: "other_vehicle"
    }
    item {
      id: 13
      name: "train"
    }
    item {
      id: 14
      name: "pedestrian"
    }
    item {
      id: 15
      name: "traffic_light_y"
    }
    item {
      id: 16
      name: "car"
    }


## Install Object Detection module from tensorflow


```python
import os
import pathlib

# Clone the tensorflow models repository if it doesn't already exist
if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  !git clone --depth 1 https://github.com/tensorflow/models

```

    Cloning into 'models'...
    remote: Enumerating objects: 2234, done.[K
    remote: Counting objects: 100% (2234/2234), done.[K
    remote: Compressing objects: 100% (1930/1930), done.[K
    remote: Total 2234 (delta 541), reused 939 (delta 279), pack-reused 0[K
    Receiving objects: 100% (2234/2234), 30.48 MiB | 18.64 MiB/s, done.
    Resolving deltas: 100% (541/541), done.



```bash
%%bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

```

    Processing /content/models/research
    Collecting avro-python3
      Downloading https://files.pythonhosted.org/packages/b2/5a/819537be46d65a01f8b8c6046ed05603fb9ef88c663b8cca840263788d58/avro-python3-1.10.0.tar.gz
    Collecting apache-beam
      Downloading https://files.pythonhosted.org/packages/ce/0e/60ce0d855df4f6b49da552dd4e5a22e10ec4766d719ef28c6c40e2ca88ba/apache_beam-2.24.0-cp36-cp36m-manylinux2010_x86_64.whl (8.6MB)
    Requirement already satisfied: pillow in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (7.0.0)
    Requirement already satisfied: lxml in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (4.2.6)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (3.2.2)
    Requirement already satisfied: Cython in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (0.29.21)
    Requirement already satisfied: contextlib2 in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (0.5.5)
    Collecting tf-slim
      Downloading https://files.pythonhosted.org/packages/02/97/b0f4a64df018ca018cc035d44f2ef08f91e2e8aa67271f6f19633a015ff7/tf_slim-1.1.0-py2.py3-none-any.whl (352kB)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (1.15.0)
    Requirement already satisfied: pycocotools in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (2.0.2)
    Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (1.4.1)
    Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (from object-detection==0.1) (1.1.2)
    Collecting tf-models-official
      Downloading https://files.pythonhosted.org/packages/5b/33/91e5e90e3e96292717245d3fe87eb3b35b07c8a2113f2da7f482040facdb/tf_models_official-2.3.0-py2.py3-none-any.whl (840kB)
    Collecting oauth2client<4,>=2.0.1
      Downloading https://files.pythonhosted.org/packages/c0/7b/bc893e35d6ca46a72faa4b9eaac25c687ce60e1fbe978993fe2de1b0ff0d/oauth2client-3.0.0.tar.gz (77kB)
    Collecting mock<3.0.0,>=1.0.1
      Downloading https://files.pythonhosted.org/packages/e6/35/f187bdf23be87092bd0f1200d43d23076cee4d0dec109f195173fd3ebc79/mock-2.0.0-py2.py3-none-any.whl (56kB)
    Requirement already satisfied: pytz>=2018.3 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (2018.9)
    Requirement already satisfied: typing-extensions<3.8.0,>=3.7.0 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (3.7.4.3)
    Requirement already satisfied: pymongo<4.0.0,>=3.8.0 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (3.11.0)
    Requirement already satisfied: grpcio<2,>=1.29.0 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (1.32.0)
    Collecting pyarrow<0.18.0,>=0.15.1; python_version >= "3.0" or platform_system != "Windows"
      Downloading https://files.pythonhosted.org/packages/ba/3f/6cac1714fff444664603f92cb9fbe91c7ae25375880158b9e9691c4584c8/pyarrow-0.17.1-cp36-cp36m-manylinux2014_x86_64.whl (63.8MB)
    Collecting requests<3.0.0,>=2.24.0
      Downloading https://files.pythonhosted.org/packages/45/1e/0c169c6a5381e241ba7404532c16a21d86ab872c9bed8bdcd4c423954103/requests-2.24.0-py2.py3-none-any.whl (61kB)
    Requirement already satisfied: python-dateutil<3,>=2.8.0 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (2.8.1)
    Collecting dill<0.3.2,>=0.3.1.1
      Downloading https://files.pythonhosted.org/packages/c7/11/345f3173809cea7f1a193bfbf02403fff250a3360e0e118a1630985e547d/dill-0.3.1.1.tar.gz (151kB)
    Collecting fastavro<0.24,>=0.21.4
      Downloading https://files.pythonhosted.org/packages/98/8e/1d62398df5569a805d956bd96df1b2c06f973e8d3f1f7489adf9c58b2824/fastavro-0.23.6-cp36-cp36m-manylinux2010_x86_64.whl (1.4MB)
    Collecting hdfs<3.0.0,>=2.1.0
      Downloading https://files.pythonhosted.org/packages/82/39/2c0879b1bcfd1f6ad078eb210d09dbce21072386a3997074ee91e60ddc5a/hdfs-2.5.8.tar.gz (41kB)
    Requirement already satisfied: numpy<2,>=1.14.3 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (1.18.5)
    Requirement already satisfied: pydot<2,>=1.2.0 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (1.3.0)
    Requirement already satisfied: protobuf<4,>=3.12.2 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (3.12.4)
    Requirement already satisfied: httplib2<0.18.0,>=0.8 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (0.17.4)
    Collecting future<1.0.0,>=0.18.2
      Downloading https://files.pythonhosted.org/packages/45/0b/38b06fd9b92dc2b68d58b75f900e97884c45bedd2ff83203d933cf5851c9/future-0.18.2.tar.gz (829kB)
    Requirement already satisfied: crcmod<2.0,>=1.7 in /usr/local/lib/python3.6/dist-packages (from apache-beam->object-detection==0.1) (1.7)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->object-detection==0.1) (2.4.7)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->object-detection==0.1) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib->object-detection==0.1) (0.10.0)
    Requirement already satisfied: absl-py>=0.2.2 in /usr/local/lib/python3.6/dist-packages (from tf-slim->object-detection==0.1) (0.10.0)
    Requirement already satisfied: setuptools>=18.0 in /usr/local/lib/python3.6/dist-packages (from pycocotools->object-detection==0.1) (50.3.0)
    Requirement already satisfied: google-cloud-bigquery>=0.31.0 in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (1.21.0)
    Requirement already satisfied: gin-config in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (0.3.0)
    Requirement already satisfied: tensorflow-hub>=0.6.0 in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (0.9.0)
    Collecting opencv-python-headless
      Downloading https://files.pythonhosted.org/packages/e2/e2/6670da2b12544858657058a5db2f088a18c56d0144bef8d178ad4734b7a3/opencv_python_headless-4.4.0.44-cp36-cp36m-manylinux2014_x86_64.whl (36.7MB)
    Collecting tensorflow-model-optimization>=0.2.1
      Downloading https://files.pythonhosted.org/packages/55/38/4fd48ea1bfcb0b6e36d949025200426fe9c3a8bfae029f0973d85518fa5a/tensorflow_model_optimization-0.5.0-py2.py3-none-any.whl (172kB)
    Requirement already satisfied: dataclasses in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (0.7)
    Requirement already satisfied: tensorflow-datasets in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (2.1.0)
    Requirement already satisfied: psutil>=5.4.3 in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (5.4.8)
    Collecting sentencepiece
      Downloading https://files.pythonhosted.org/packages/d4/a4/d0a884c4300004a78cca907a6ff9a5e9fe4f090f5d95ab341c53d28cbc58/sentencepiece-0.1.91-cp36-cp36m-manylinux1_x86_64.whl (1.1MB)
    Requirement already satisfied: tensorflow>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (2.3.0)
    Requirement already satisfied: google-api-python-client>=1.6.7 in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (1.7.12)
    Collecting py-cpuinfo>=3.3.0
      Downloading https://files.pythonhosted.org/packages/f6/f5/8e6e85ce2e9f6e05040cf0d4e26f43a4718bcc4bce988b433276d4b1a5c1/py-cpuinfo-7.0.0.tar.gz (95kB)
    Requirement already satisfied: kaggle>=1.3.9 in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (1.5.8)
    Requirement already satisfied: tensorflow-addons in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (0.8.3)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.6/dist-packages (from tf-models-official->object-detection==0.1) (3.13)
    Requirement already satisfied: pyasn1>=0.1.7 in /usr/local/lib/python3.6/dist-packages (from oauth2client<4,>=2.0.1->apache-beam->object-detection==0.1) (0.4.8)
    Requirement already satisfied: pyasn1-modules>=0.0.5 in /usr/local/lib/python3.6/dist-packages (from oauth2client<4,>=2.0.1->apache-beam->object-detection==0.1) (0.2.8)
    Requirement already satisfied: rsa>=3.1.4 in /usr/local/lib/python3.6/dist-packages (from oauth2client<4,>=2.0.1->apache-beam->object-detection==0.1) (4.6)
    Collecting pbr>=0.11
      Downloading https://files.pythonhosted.org/packages/c1/a3/d439f338aa90edd5ad9096cd56564b44882182150e92148eb14ceb7488ba/pbr-5.5.0-py2.py3-none-any.whl (106kB)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.24.0->apache-beam->object-detection==0.1) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.24.0->apache-beam->object-detection==0.1) (2020.6.20)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.24.0->apache-beam->object-detection==0.1) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.24.0->apache-beam->object-detection==0.1) (3.0.4)
    Requirement already satisfied: docopt in /usr/local/lib/python3.6/dist-packages (from hdfs<3.0.0,>=2.1.0->apache-beam->object-detection==0.1) (0.6.2)
    Requirement already satisfied: google-cloud-core<2.0dev,>=1.0.3 in /usr/local/lib/python3.6/dist-packages (from google-cloud-bigquery>=0.31.0->tf-models-official->object-detection==0.1) (1.0.3)
    Requirement already satisfied: google-resumable-media!=0.4.0,<0.5.0dev,>=0.3.1 in /usr/local/lib/python3.6/dist-packages (from google-cloud-bigquery>=0.31.0->tf-models-official->object-detection==0.1) (0.4.1)
    Requirement already satisfied: dm-tree~=0.1.1 in /usr/local/lib/python3.6/dist-packages (from tensorflow-model-optimization>=0.2.1->tf-models-official->object-detection==0.1) (0.1.5)
    Requirement already satisfied: tensorflow-metadata in /usr/local/lib/python3.6/dist-packages (from tensorflow-datasets->tf-models-official->object-detection==0.1) (0.24.0)
    Requirement already satisfied: termcolor in /usr/local/lib/python3.6/dist-packages (from tensorflow-datasets->tf-models-official->object-detection==0.1) (1.1.0)
    Requirement already satisfied: wrapt in /usr/local/lib/python3.6/dist-packages (from tensorflow-datasets->tf-models-official->object-detection==0.1) (1.12.1)
    Requirement already satisfied: attrs>=18.1.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow-datasets->tf-models-official->object-detection==0.1) (20.2.0)
    Requirement already satisfied: promise in /usr/local/lib/python3.6/dist-packages (from tensorflow-datasets->tf-models-official->object-detection==0.1) (2.3)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from tensorflow-datasets->tf-models-official->object-detection==0.1) (4.41.1)
    Requirement already satisfied: google-pasta>=0.1.8 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (0.2.0)
    Requirement already satisfied: h5py<2.11.0,>=2.10.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (2.10.0)
    Requirement already satisfied: keras-preprocessing<1.2,>=1.1.1 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (1.1.2)
    Requirement already satisfied: tensorflow-estimator<2.4.0,>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (2.3.0)
    Requirement already satisfied: astunparse==1.6.3 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (1.6.3)
    Requirement already satisfied: gast==0.3.3 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (0.3.3)
    Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (0.35.1)
    Requirement already satisfied: tensorboard<3,>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (2.3.0)
    Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.6/dist-packages (from tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (3.3.0)
    Requirement already satisfied: uritemplate<4dev,>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from google-api-python-client>=1.6.7->tf-models-official->object-detection==0.1) (3.0.1)
    Requirement already satisfied: google-auth>=1.4.1 in /usr/local/lib/python3.6/dist-packages (from google-api-python-client>=1.6.7->tf-models-official->object-detection==0.1) (1.17.2)
    Requirement already satisfied: google-auth-httplib2>=0.0.3 in /usr/local/lib/python3.6/dist-packages (from google-api-python-client>=1.6.7->tf-models-official->object-detection==0.1) (0.0.4)
    Requirement already satisfied: python-slugify in /usr/local/lib/python3.6/dist-packages (from kaggle>=1.3.9->tf-models-official->object-detection==0.1) (4.0.1)
    Requirement already satisfied: slugify in /usr/local/lib/python3.6/dist-packages (from kaggle>=1.3.9->tf-models-official->object-detection==0.1) (0.0.1)
    Requirement already satisfied: typeguard in /usr/local/lib/python3.6/dist-packages (from tensorflow-addons->tf-models-official->object-detection==0.1) (2.7.1)
    Requirement already satisfied: google-api-core<2.0.0dev,>=1.14.0 in /usr/local/lib/python3.6/dist-packages (from google-cloud-core<2.0dev,>=1.0.3->google-cloud-bigquery>=0.31.0->tf-models-official->object-detection==0.1) (1.16.0)
    Requirement already satisfied: googleapis-common-protos<2,>=1.52.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow-metadata->tensorflow-datasets->tf-models-official->object-detection==0.1) (1.52.0)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (3.2.2)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (0.4.1)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (1.7.0)
    Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.6/dist-packages (from tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (1.0.1)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.6/dist-packages (from google-auth>=1.4.1->google-api-python-client>=1.6.7->tf-models-official->object-detection==0.1) (4.1.1)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.6/dist-packages (from python-slugify->kaggle>=1.3.9->tf-models-official->object-detection==0.1) (1.3)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from markdown>=2.6.8->tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (2.0.0)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (1.3.0)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata; python_version < "3.8"->markdown>=2.6.8->tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (3.2.0)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<3,>=2.3.0->tensorflow>=2.3.0->tf-models-official->object-detection==0.1) (3.1.0)
    Building wheels for collected packages: object-detection, avro-python3, oauth2client, dill, hdfs, future, py-cpuinfo
      Building wheel for object-detection (setup.py): started
      Building wheel for object-detection (setup.py): finished with status 'done'
      Created wheel for object-detection: filename=object_detection-0.1-cp36-none-any.whl size=1580433 sha256=5b1c3354fdfeee6b75253e6953768e1bdf145d961969b45e1e2d963defe82591
      Stored in directory: /tmp/pip-ephem-wheel-cache-b9lmecna/wheels/94/49/4b/39b051683087a22ef7e80ec52152a27249d1a644ccf4e442ea
      Building wheel for avro-python3 (setup.py): started
      Building wheel for avro-python3 (setup.py): finished with status 'done'
      Created wheel for avro-python3: filename=avro_python3-1.10.0-cp36-none-any.whl size=43735 sha256=1915f4167cf553e2337241e4a5808f29e5aa7dbd2b03f1a5df5cb5bea9647605
      Stored in directory: /root/.cache/pip/wheels/3f/15/cd/fe4ec8b88c130393464703ee8111e2cddebdc40e1b59ea85e9
      Building wheel for oauth2client (setup.py): started
      Building wheel for oauth2client (setup.py): finished with status 'done'
      Created wheel for oauth2client: filename=oauth2client-3.0.0-cp36-none-any.whl size=106382 sha256=bf02908fc9583b1bf46870fe32d18e6d45d5beca729ca2509bcbe84d35564442
      Stored in directory: /root/.cache/pip/wheels/48/f7/87/b932f09c6335dbcf45d916937105a372ab14f353a9ca431d7d
      Building wheel for dill (setup.py): started
      Building wheel for dill (setup.py): finished with status 'done'
      Created wheel for dill: filename=dill-0.3.1.1-cp36-none-any.whl size=78532 sha256=4081e92a43122e63842d946fc5517ef695744100b8310e6ec9f607289a2cdf2b
      Stored in directory: /root/.cache/pip/wheels/59/b1/91/f02e76c732915c4015ab4010f3015469866c1eb9b14058d8e7
      Building wheel for hdfs (setup.py): started
      Building wheel for hdfs (setup.py): finished with status 'done'
      Created wheel for hdfs: filename=hdfs-2.5.8-cp36-none-any.whl size=33213 sha256=cdb4533e4db79f853db5dd25d432e5336499c7e3d8e02b9aa6d0c4adf733d7f0
      Stored in directory: /root/.cache/pip/wheels/fe/a7/05/23e3699975fc20f8a30e00ac1e515ab8c61168e982abe4ce70
      Building wheel for future (setup.py): started
      Building wheel for future (setup.py): finished with status 'done'
      Created wheel for future: filename=future-0.18.2-cp36-none-any.whl size=491057 sha256=6b5fa4add6a4b4bad2df3ac47ac2f7c5617af2e09dc9573d34d17289f2ee8a61
      Stored in directory: /root/.cache/pip/wheels/8b/99/a0/81daf51dcd359a9377b110a8a886b3895921802d2fc1b2397e
      Building wheel for py-cpuinfo (setup.py): started
      Building wheel for py-cpuinfo (setup.py): finished with status 'done'
      Created wheel for py-cpuinfo: filename=py_cpuinfo-7.0.0-cp36-none-any.whl size=20071 sha256=3a79c6d359ae8664a85fe97232e8aa67b9e877a8df80cd49301af00b84feb230
      Stored in directory: /root/.cache/pip/wheels/f1/93/7b/127daf0c3a5a49feb2fecd468d508067c733fba5192f726ad1
    Successfully built object-detection avro-python3 oauth2client dill hdfs future py-cpuinfo
    Installing collected packages: avro-python3, oauth2client, pbr, mock, pyarrow, requests, dill, fastavro, hdfs, future, apache-beam, tf-slim, opencv-python-headless, tensorflow-model-optimization, sentencepiece, py-cpuinfo, tf-models-official, object-detection
      Found existing installation: oauth2client 4.1.3
        Uninstalling oauth2client-4.1.3:
          Successfully uninstalled oauth2client-4.1.3
      Found existing installation: pyarrow 0.14.1
        Uninstalling pyarrow-0.14.1:
          Successfully uninstalled pyarrow-0.14.1
      Found existing installation: requests 2.23.0
        Uninstalling requests-2.23.0:
          Successfully uninstalled requests-2.23.0
      Found existing installation: dill 0.3.2
        Uninstalling dill-0.3.2:
          Successfully uninstalled dill-0.3.2
      Found existing installation: future 0.16.0
        Uninstalling future-0.16.0:
          Successfully uninstalled future-0.16.0
    Successfully installed apache-beam-2.24.0 avro-python3-1.10.0 dill-0.3.1.1 fastavro-0.23.6 future-0.18.2 hdfs-2.5.8 mock-2.0.0 oauth2client-3.0.0 object-detection-0.1 opencv-python-headless-4.4.0.44 pbr-5.5.0 py-cpuinfo-7.0.0 pyarrow-0.17.1 requests-2.24.0 sentencepiece-0.1.91 tensorflow-model-optimization-0.5.0 tf-models-official-2.3.0 tf-slim-1.1.0


    ERROR: pydrive 1.3.1 has requirement oauth2client>=4.0.0, but you'll have oauth2client 3.0.0 which is incompatible.
    ERROR: multiprocess 0.70.10 has requirement dill>=0.3.2, but you'll have dill 0.3.1.1 which is incompatible.
    ERROR: google-colab 1.0.0 has requirement requests~=2.23.0, but you'll have requests 2.24.0 which is incompatible.
    ERROR: datascience 0.10.6 has requirement folium==0.2.1, but you'll have folium 0.8.3 which is incompatible.
    ERROR: apache-beam 2.24.0 has requirement avro-python3!=1.9.2,<1.10.0,>=1.8.1; python_version >= "3.0", but you'll have avro-python3 1.10.0 which is incompatible.


# Build the .TFrecord files for validation and training

## This code was heavily modified from online version to make it compatible with TensorFlow v2 as well as make it able to parse .json labels instead of .csv. 



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
train_labels_path = 'bdd100k_data/labels/det_v2_train_release.json'
train_image_path = 'bdd100k_data/images'
train_record_path = 'train.record'

test_size = None # int num of images (None: Convert all images)
test_labels_path = 'bdd100k_data/labels/det_v2_val_release.json'
test_image_path = 'bdd100k_data/images'
test_record_path = 'test.record'

# We pass in the exact path that contains images for O(1) image finding operation
for labels_path, images_path, record_name, limit in [(test_labels_path, test_image_path, test_record_path, test_size), (train_labels_path, train_image_path, train_record_path, train_size)]:
  create_records(labels_path=labels_path, images_path=images_path, output_record_name=record_name, limit=limit)
```

    test.record processed 10000/10000 images
    Successfully created the TFRecords: test.record
    train.record processed 69863/69863 images
    Successfully created the TFRecords: train.record


# I use faster_rcnn_resnet50_v1_640x640_coco17_tpu-8  which can be found in Tensorflow v2 Model Zoo.


```python
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
!tar -xf faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
!rm faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
```

    --2020-10-11 23:07:49--  http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
    Resolving download.tensorflow.org (download.tensorflow.org)... 108.177.126.128, 2a00:1450:4013:c01::80
    Connecting to download.tensorflow.org (download.tensorflow.org)|108.177.126.128|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 211996178 (202M) [application/x-tar]
    Saving to: ‘faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz’
    
    faster_rcnn_resnet5 100%[===================>] 202.17M  66.8MB/s    in 3.0s    
    
    2020-10-11 23:07:53 (66.8 MB/s) - ‘faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz’ saved [211996178/211996178]
    


# Build some of the pipeline config file's parameters

# NOTE: This code was used for the inital development of the config file. The pipeline config file is later modified for a different optimizer, num_steps and eval config.


```python
num_steps = 40000 # initial number of steps

model_name = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'
pretrained_checkpoint = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz'
base_pipeline_file = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config'
batch_size = 4

pipeline_fname = './'+ model_name + '/pipeline.config'
fine_tune_checkpoint = './' + model_name + '/checkpoint/ckpt-0'
label_map_pbtxt_fname = 'label_map.pbtxt'

num_classes = len(classes)
train_record_fname = "train.record"
test_record_fname = "test.record"
```


```python
#write custom configuration file by slotting our dataset, model checkpoint, and training parameters into the base pipeline file

import re
print('writing custom configuration file')

with open(pipeline_fname) as f:
    s = f.read()

with open('pipeline_file.config', 'w') as f:
    
    # fine_tune_checkpoint
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
    
    # tfrecord files train and test.
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

    # label_map_path
    s = re.sub(
        'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

    # Set training batch_size.
    s = re.sub('batch_size: [0-9]+',
               'batch_size: {}'.format(batch_size), s)

    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+',
               'num_steps: {}'.format(num_steps), s)
    
    # Set number of classes num_classes.
    s = re.sub('num_classes: [0-9]+',
               'num_classes: {}'.format(num_classes), s)
    
    #fine-tune checkpoint type
    s = re.sub(
        'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
        
    f.write(s)


```

    writing custom configuration file



## I've attempted a couple of different optimizers. First, the default momentum optimizer with default learning rates/steps. This optimizer did not yield good training scores after 50k - 90k steps. The loss was fluctuating between high values and low values.

## Then, I tried adam optimizer with manual learning rate as follows:



```
    adam_optimizer: {
      learning_rate: {
        manual_step_learning_rate {
          initial_learning_rate: .0002
          schedule {
            step: 50000
            learning_rate: .0001
          }
          schedule {
            step: 90000
            learning_rate: .00008
          }
          schedule {
            step: 120000
            learning_rate: .00004
          }
        }
      }
      # momentum_optimizer_value: 0.9
    }
```

## I found that the above optimizer was a little bit more stable and I saw a continous decrease in loss from ~1.8 to ~0.5.





```python
!cat pipeline_file.config
```

    # Faster R-CNN with Resnet-50 (v1)
    # Trained on COCO, initialized from Imagenet classification checkpoint
    
    # Achieves -- mAP on COCO14 minival dataset.
    
    # This config is TPU compatible.
    
    model {
      faster_rcnn {
        num_classes: 16
        image_resizer {
          keep_aspect_ratio_resizer {
            min_dimension: 640
            max_dimension: 640
            pad_to_max_dimension: true
          }
        }
        feature_extractor {
          type: 'faster_rcnn_resnet50_keras'
          batch_norm_trainable: true
        }
        first_stage_anchor_generator {
          grid_anchor_generator {
            scales: [0.25, 0.5, 1.0, 2.0]
            aspect_ratios: [0.5, 1.0, 2.0]
            height_stride: 16
            width_stride: 16
          }
        }
        first_stage_box_predictor_conv_hyperparams {
          op: CONV
          regularizer {
            l2_regularizer {
              weight: 0.0
            }
          }
          initializer {
            truncated_normal_initializer {
              stddev: 0.01
            }
          }
        }
        first_stage_nms_score_threshold: 0.0
        first_stage_nms_iou_threshold: 0.7
        first_stage_max_proposals: 300
        first_stage_localization_loss_weight: 2.0
        first_stage_objectness_loss_weight: 1.0
        initial_crop_size: 14
        maxpool_kernel_size: 2
        maxpool_stride: 2
        second_stage_box_predictor {
          mask_rcnn_box_predictor {
            use_dropout: false
            dropout_keep_probability: 1.0
            fc_hyperparams {
              op: FC
              regularizer {
                l2_regularizer {
                  weight: 0.0
                }
              }
              initializer {
                variance_scaling_initializer {
                  factor: 1.0
                  uniform: true
                  mode: FAN_AVG
                }
              }
            }
            share_box_across_classes: true
          }
        }
        second_stage_post_processing {
          batch_non_max_suppression {
            score_threshold: 0.0
            iou_threshold: 0.6
            max_detections_per_class: 100
            max_total_detections: 300
          }
          score_converter: SOFTMAX
        }
        second_stage_localization_loss_weight: 2.0
        second_stage_classification_loss_weight: 1.0
        use_static_shapes: true
        use_matmul_crop_and_resize: true
        clip_anchors_to_image: true
        use_static_balanced_label_sampler: true
        use_matmul_gather_in_matcher: true
      }
    }
    
    train_config: {
      batch_size: 4
      sync_replicas: true
      startup_delay_steps: 0
      replicas_to_aggregate: 8
      num_steps: 40000
    optimizer {
        adam_optimizer: {
          learning_rate: {
            manual_step_learning_rate {
              initial_learning_rate: .0002
              schedule {
                step: 5000
                learning_rate: .0001
              }
              schedule {
                step: 17000
                learning_rate: .00008
              }
              schedule {
                step: 30000
                learning_rate: .00004
              }
            }
          }
          # momentum_optimizer_value: 0.9
        }
        use_moving_average: false
      }
      fine_tune_checkpoint_version: V2
      fine_tune_checkpoint: "./faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/checkpoint/ckpt-0"
      fine_tune_checkpoint_type: "detection"
      data_augmentation_options {
        random_horizontal_flip {
        }
      }
    
      max_number_of_boxes: 100
      unpad_groundtruth_tensors: false
      use_bfloat16: true  # works only on TPUs
    }
    
    train_input_reader: {
      label_map_path: "label_map.pbtxt"
      tf_record_input_reader {
        input_path: "train.record"
      }
    }
    
    eval_config: {
      metrics_set: "coco_detection_metrics"
      use_moving_averages: false
      batch_size: 4;
    }
    
    eval_input_reader: {
      label_map_path: "label_map.pbtxt"
      shuffle: false
      num_epochs: 1
      tf_record_input_reader {
        input_path: "test.record"
      }
    }


# Start training the model with above parameters


```python
!python models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={'./pipeline_file.config'} \
    --model_dir={'./myoutputmodel'} \
    --alsologtostderr
```

    2020-10-09 05:47:52.953828: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-09 05:47:55.445313: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
    2020-10-09 05:47:55.451672: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 05:47:55.452210: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
    coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.75GiB deviceMemoryBandwidth: 836.37GiB/s
    2020-10-09 05:47:55.452243: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-09 05:47:55.454488: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2020-10-09 05:47:55.456639: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
    2020-10-09 05:47:55.457008: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
    2020-10-09 05:47:55.459363: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
    2020-10-09 05:47:55.460730: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
    2020-10-09 05:47:55.465340: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    2020-10-09 05:47:55.465525: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 05:47:55.466077: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 05:47:55.466543: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
    2020-10-09 05:47:55.466991: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX512F
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2020-10-09 05:47:55.473396: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2000165000 Hz
    2020-10-09 05:47:55.473685: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x20f6f40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-10-09 05:47:55.473719: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-10-09 05:47:55.566823: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 05:47:55.567833: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x20f7100 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2020-10-09 05:47:55.567889: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
    2020-10-09 05:47:55.568177: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 05:47:55.568724: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
    coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.75GiB deviceMemoryBandwidth: 836.37GiB/s
    2020-10-09 05:47:55.568775: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-09 05:47:55.568842: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2020-10-09 05:47:55.568876: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
    2020-10-09 05:47:55.568935: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
    2020-10-09 05:47:55.568963: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
    2020-10-09 05:47:55.568992: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
    2020-10-09 05:47:55.569036: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    2020-10-09 05:47:55.569150: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 05:47:55.569947: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 05:47:55.570685: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
    2020-10-09 05:47:55.570771: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-09 05:47:56.311923: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-10-09 05:47:56.311995: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0 
    2020-10-09 05:47:56.312006: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N 
    2020-10-09 05:47:56.312264: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 05:47:56.312866: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 05:47:56.313371: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
    2020-10-09 05:47:56.313414: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 9621 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0)
    INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0',)
    I1009 05:47:56.317520 139781364750208 mirrored_strategy.py:341] Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0',)
    INFO:tensorflow:Maybe overwriting train_steps: None
    I1009 05:47:56.322495 139781364750208 config_util.py:552] Maybe overwriting train_steps: None
    INFO:tensorflow:Maybe overwriting use_bfloat16: False
    I1009 05:47:56.322676 139781364750208 config_util.py:552] Maybe overwriting use_bfloat16: False
    INFO:tensorflow:Reading unweighted datasets: ['train.record']
    I1009 05:47:56.386153 139781364750208 dataset_builder.py:148] Reading unweighted datasets: ['train.record']
    INFO:tensorflow:Reading record datasets for input file: ['train.record']
    I1009 05:47:56.387457 139781364750208 dataset_builder.py:77] Reading record datasets for input file: ['train.record']
    INFO:tensorflow:Number of filenames to read: 1
    I1009 05:47:56.387642 139781364750208 dataset_builder.py:78] Number of filenames to read: 1
    WARNING:tensorflow:num_readers has been reduced to 1 to match input file shards.
    W1009 05:47:56.387714 139781364750208 dataset_builder.py:86] num_readers has been reduced to 1 to match input file shards.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:103: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.experimental_deterministic`.
    W1009 05:47:56.392982 139781364750208 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:103: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.experimental_deterministic`.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:222: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.map()
    W1009 05:47:56.431547 139781364750208 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:222: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.map()
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
    W1009 05:48:03.876277 139781364750208 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/inputs.py:259: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    W1009 05:48:06.938496 139781364750208 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/inputs.py:259: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib_v2.py:355: set_learning_phase (from tensorflow.python.keras.backend) is deprecated and will be removed after 2020-10-11.
    Instructions for updating:
    Simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
    W1009 05:48:14.829178 139777676863232 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib_v2.py:355: set_learning_phase (from tensorflow.python.keras.backend) is deprecated and will be removed after 2020-10-11.
    Instructions for updating:
    Simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
    INFO:tensorflow:depth of additional conv before box predictor: 0
    I1009 05:48:20.049914 139777676863232 convolutional_keras_box_predictor.py:154] depth of additional conv before box predictor: 0
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/model_util.py:57: Tensor.experimental_ref (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use ref() instead.
    W1009 05:48:28.971289 139777676863232 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/model_util.py:57: Tensor.experimental_ref (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use ref() instead.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    W1009 05:48:35.029056 139777676863232 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    2020-10-09 05:48:51.088912: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2020-10-09 05:48:51.453768: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._groundtruth_lists
    W1009 05:48:57.908310 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._groundtruth_lists
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv
    W1009 05:48:57.908734 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor
    W1009 05:48:57.908839 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._maxpool_layer
    W1009 05:48:57.908923 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._maxpool_layer
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor
    W1009 05:48:57.909000 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._batched_prediction_tensor_names
    W1009 05:48:57.909072 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._batched_prediction_tensor_names
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model.endpoints
    W1009 05:48:57.909144 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model.endpoints
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0
    W1009 05:48:57.909214 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-1
    W1009 05:48:57.909289 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-1
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-2
    W1009 05:48:57.909373 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-2
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads
    W1009 05:48:57.909449 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._sorted_head_names
    W1009 05:48:57.909520 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._sorted_head_names
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._shared_nets
    W1009 05:48:57.909590 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._shared_nets
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head
    W1009 05:48:57.909658 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head
    W1009 05:48:57.909734 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._third_stage_heads
    W1009 05:48:57.909804 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._third_stage_heads
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0._inbound_nodes
    W1009 05:48:57.909890 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0._inbound_nodes
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0.kernel
    W1009 05:48:57.909961 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0.kernel
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0.bias
    W1009 05:48:57.910033 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0.bias
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-1._inbound_nodes
    W1009 05:48:57.910103 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-1._inbound_nodes
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-2._inbound_nodes
    W1009 05:48:57.910173 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-2._inbound_nodes
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings
    W1009 05:48:57.910242 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background
    W1009 05:48:57.910312 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._shared_nets.0
    W1009 05:48:57.910395 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._shared_nets.0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers
    W1009 05:48:57.910465 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers
    W1009 05:48:57.910534 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0
    W1009 05:48:57.910661 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0
    W1009 05:48:57.910737 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.0
    W1009 05:48:57.910812 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.1
    W1009 05:48:57.910883 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.1
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.2
    W1009 05:48:57.910952 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.2
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.0
    W1009 05:48:57.911021 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.1
    W1009 05:48:57.911091 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.1
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.2
    W1009 05:48:57.911160 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.2
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers
    W1009 05:48:57.911231 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers
    W1009 05:48:57.911302 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.1.kernel
    W1009 05:48:57.911391 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.1.kernel
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.1.bias
    W1009 05:48:57.911464 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.1.bias
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.1.kernel
    W1009 05:48:57.911533 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.1.kernel
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.1.bias
    W1009 05:48:57.911603 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.1.bias
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers.0
    W1009 05:48:57.911673 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers.0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers.0
    W1009 05:48:57.911748 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers.0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers.0.kernel
    W1009 05:48:57.911818 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers.0.kernel
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers.0.bias
    W1009 05:48:57.911888 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers.0.bias
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers.0.kernel
    W1009 05:48:57.911958 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers.0.kernel
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers.0.bias
    W1009 05:48:57.912033 139781364750208 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers.0.bias
    WARNING:tensorflow:A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.
    W1009 05:48:57.912106 139781364750208 util.py:158] A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1009 05:48:58.397938 139781364750208 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1009 05:48:58.399501 139781364750208 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1009 05:48:58.401373 139781364750208 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1009 05:48:58.402481 139781364750208 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1009 05:48:58.404647 139781364750208 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1009 05:48:58.405494 139781364750208 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1009 05:48:58.407466 139781364750208 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1009 05:48:58.408276 139781364750208 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1009 05:48:58.409665 139781364750208 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1009 05:48:58.410495 139781364750208 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/deprecation.py:574: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Use fn_output_signature instead
    W1009 05:49:09.420664 139777685255936 deprecation.py:506] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/deprecation.py:574: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Use fn_output_signature instead
    2020-10-09 05:49:29.164163: W tensorflow/core/common_runtime/bfc_allocator.cc:246] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.14GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
    INFO:tensorflow:Step 100 per-step time 0.260s loss=1.553
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
    I1009 05:52:02.484522 139781364750208 model_lib_v2.py:652] Step 600 per-step time 0.241s loss=1.397
    INFO:tensorflow:Step 700 per-step time 0.245s loss=1.491
    I1009 05:52:27.949731 139781364750208 model_lib_v2.py:652] Step 700 per-step time 0.245s loss=1.491
    INFO:tensorflow:Step 800 per-step time 0.260s loss=1.498
    I1009 05:52:53.430289 139781364750208 model_lib_v2.py:652] Step 800 per-step time 0.260s loss=1.498
    INFO:tensorflow:Step 900 per-step time 0.251s loss=1.572
    I1009 05:53:18.978099 139781364750208 model_lib_v2.py:652] Step 900 per-step time 0.251s loss=1.572
    INFO:tensorflow:Step 1000 per-step time 0.269s loss=1.218
    I1009 05:53:44.199573 139781364750208 model_lib_v2.py:652] Step 1000 per-step time 0.269s loss=1.218
    INFO:tensorflow:Step 1100 per-step time 0.258s loss=1.331
    I1009 05:54:10.724823 139781364750208 model_lib_v2.py:652] Step 1100 per-step time 0.258s loss=1.331
    INFO:tensorflow:Step 1200 per-step time 0.278s loss=0.994
    I1009 05:54:36.385244 139781364750208 model_lib_v2.py:652] Step 1200 per-step time 0.278s loss=0.994
    INFO:tensorflow:Step 1300 per-step time 0.247s loss=1.272
    I1009 05:55:02.069950 139781364750208 model_lib_v2.py:652] Step 1300 per-step time 0.247s loss=1.272
    INFO:tensorflow:Step 1400 per-step time 0.244s loss=1.658
    I1009 05:55:27.579403 139781364750208 model_lib_v2.py:652] Step 1400 per-step time 0.244s loss=1.658
    INFO:tensorflow:Step 1500 per-step time 0.243s loss=1.684
    I1009 05:55:53.054051 139781364750208 model_lib_v2.py:652] Step 1500 per-step time 0.243s loss=1.684
    INFO:tensorflow:Step 1600 per-step time 0.253s loss=1.370
    I1009 05:56:18.824827 139781364750208 model_lib_v2.py:652] Step 1600 per-step time 0.253s loss=1.370
    INFO:tensorflow:Step 1700 per-step time 0.266s loss=1.004
    I1009 05:56:44.469465 139781364750208 model_lib_v2.py:652] Step 1700 per-step time 0.266s loss=1.004
    INFO:tensorflow:Step 1800 per-step time 0.247s loss=1.090
    I1009 05:57:10.033257 139781364750208 model_lib_v2.py:652] Step 1800 per-step time 0.247s loss=1.090
    INFO:tensorflow:Step 1900 per-step time 0.265s loss=0.632
    I1009 05:57:35.561442 139781364750208 model_lib_v2.py:652] Step 1900 per-step time 0.265s loss=0.632
    INFO:tensorflow:Step 2000 per-step time 0.251s loss=1.349
    I1009 05:58:01.241011 139781364750208 model_lib_v2.py:652] Step 2000 per-step time 0.251s loss=1.349
    INFO:tensorflow:Step 2100 per-step time 0.262s loss=1.476
    I1009 05:58:27.692660 139781364750208 model_lib_v2.py:652] Step 2100 per-step time 0.262s loss=1.476
    INFO:tensorflow:Step 2200 per-step time 0.264s loss=1.090
    I1009 05:58:53.180758 139781364750208 model_lib_v2.py:652] Step 2200 per-step time 0.264s loss=1.090
    INFO:tensorflow:Step 2300 per-step time 0.258s loss=1.300
    I1009 05:59:18.735072 139781364750208 model_lib_v2.py:652] Step 2300 per-step time 0.258s loss=1.300
    INFO:tensorflow:Step 2400 per-step time 0.249s loss=1.374
    I1009 05:59:44.417292 139781364750208 model_lib_v2.py:652] Step 2400 per-step time 0.249s loss=1.374
    INFO:tensorflow:Step 2500 per-step time 0.266s loss=0.940
    I1009 06:00:10.135698 139781364750208 model_lib_v2.py:652] Step 2500 per-step time 0.266s loss=0.940
    INFO:tensorflow:Step 2600 per-step time 0.245s loss=0.960
    I1009 06:00:35.509999 139781364750208 model_lib_v2.py:652] Step 2600 per-step time 0.245s loss=0.960
    INFO:tensorflow:Step 2700 per-step time 0.269s loss=1.454
    I1009 06:01:01.057427 139781364750208 model_lib_v2.py:652] Step 2700 per-step time 0.269s loss=1.454
    INFO:tensorflow:Step 2800 per-step time 0.259s loss=1.301
    I1009 06:01:26.763753 139781364750208 model_lib_v2.py:652] Step 2800 per-step time 0.259s loss=1.301
    INFO:tensorflow:Step 2900 per-step time 0.254s loss=0.855
    I1009 06:01:52.394306 139781364750208 model_lib_v2.py:652] Step 2900 per-step time 0.254s loss=0.855
    INFO:tensorflow:Step 3000 per-step time 0.254s loss=1.538
    I1009 06:02:17.996537 139781364750208 model_lib_v2.py:652] Step 3000 per-step time 0.254s loss=1.538
    INFO:tensorflow:Step 3100 per-step time 0.253s loss=1.431
    I1009 06:02:44.286414 139781364750208 model_lib_v2.py:652] Step 3100 per-step time 0.253s loss=1.431
    INFO:tensorflow:Step 3200 per-step time 0.249s loss=1.111
    I1009 06:03:09.699565 139781364750208 model_lib_v2.py:652] Step 3200 per-step time 0.249s loss=1.111
    INFO:tensorflow:Step 3300 per-step time 0.258s loss=1.049
    I1009 06:03:35.139602 139781364750208 model_lib_v2.py:652] Step 3300 per-step time 0.258s loss=1.049
    INFO:tensorflow:Step 3400 per-step time 0.259s loss=1.153
    I1009 06:04:00.643924 139781364750208 model_lib_v2.py:652] Step 3400 per-step time 0.259s loss=1.153
    INFO:tensorflow:Step 3500 per-step time 0.249s loss=1.061
    I1009 06:04:25.994251 139781364750208 model_lib_v2.py:652] Step 3500 per-step time 0.249s loss=1.061
    INFO:tensorflow:Step 3600 per-step time 0.240s loss=0.963
    I1009 06:04:51.408278 139781364750208 model_lib_v2.py:652] Step 3600 per-step time 0.240s loss=0.963
    INFO:tensorflow:Step 3700 per-step time 0.258s loss=1.372
    I1009 06:05:16.878058 139781364750208 model_lib_v2.py:652] Step 3700 per-step time 0.258s loss=1.372
    INFO:tensorflow:Step 3800 per-step time 0.253s loss=0.996
    I1009 06:05:42.288877 139781364750208 model_lib_v2.py:652] Step 3800 per-step time 0.253s loss=0.996
    INFO:tensorflow:Step 3900 per-step time 0.254s loss=1.180
    I1009 06:06:07.598254 139781364750208 model_lib_v2.py:652] Step 3900 per-step time 0.254s loss=1.180
    INFO:tensorflow:Step 4000 per-step time 0.251s loss=1.250
    I1009 06:06:33.337783 139781364750208 model_lib_v2.py:652] Step 4000 per-step time 0.251s loss=1.250
    INFO:tensorflow:Step 4100 per-step time 0.250s loss=0.962
    I1009 06:06:59.897112 139781364750208 model_lib_v2.py:652] Step 4100 per-step time 0.250s loss=0.962
    INFO:tensorflow:Step 4200 per-step time 0.244s loss=1.418
    I1009 06:07:25.451241 139781364750208 model_lib_v2.py:652] Step 4200 per-step time 0.244s loss=1.418
    INFO:tensorflow:Step 4300 per-step time 0.256s loss=1.200
    I1009 06:07:50.930423 139781364750208 model_lib_v2.py:652] Step 4300 per-step time 0.256s loss=1.200
    INFO:tensorflow:Step 4400 per-step time 0.256s loss=1.583
    I1009 06:08:16.422442 139781364750208 model_lib_v2.py:652] Step 4400 per-step time 0.256s loss=1.583
    INFO:tensorflow:Step 4500 per-step time 0.264s loss=1.218
    I1009 06:08:41.924167 139781364750208 model_lib_v2.py:652] Step 4500 per-step time 0.264s loss=1.218
    INFO:tensorflow:Step 4600 per-step time 0.259s loss=1.091
    I1009 06:09:07.403363 139781364750208 model_lib_v2.py:652] Step 4600 per-step time 0.259s loss=1.091
    INFO:tensorflow:Step 4700 per-step time 0.259s loss=1.460
    I1009 06:09:32.840576 139781364750208 model_lib_v2.py:652] Step 4700 per-step time 0.259s loss=1.460
    INFO:tensorflow:Step 4800 per-step time 0.248s loss=0.995
    I1009 06:09:58.385951 139781364750208 model_lib_v2.py:652] Step 4800 per-step time 0.248s loss=0.995
    INFO:tensorflow:Step 4900 per-step time 0.256s loss=1.400
    I1009 06:10:23.789413 139781364750208 model_lib_v2.py:652] Step 4900 per-step time 0.256s loss=1.400
    INFO:tensorflow:Step 5000 per-step time 0.251s loss=0.795
    I1009 06:10:49.244764 139781364750208 model_lib_v2.py:652] Step 5000 per-step time 0.251s loss=0.795
    INFO:tensorflow:Step 5100 per-step time 0.240s loss=1.285
    I1009 06:11:15.775088 139781364750208 model_lib_v2.py:652] Step 5100 per-step time 0.240s loss=1.285
    INFO:tensorflow:Step 5200 per-step time 0.268s loss=1.287
    I1009 06:11:41.417882 139781364750208 model_lib_v2.py:652] Step 5200 per-step time 0.268s loss=1.287
    INFO:tensorflow:Step 5300 per-step time 0.253s loss=0.701
    I1009 06:12:06.855992 139781364750208 model_lib_v2.py:652] Step 5300 per-step time 0.253s loss=0.701
    INFO:tensorflow:Step 5400 per-step time 0.256s loss=1.025
    I1009 06:12:32.430061 139781364750208 model_lib_v2.py:652] Step 5400 per-step time 0.256s loss=1.025
    INFO:tensorflow:Step 5500 per-step time 0.256s loss=1.085
    I1009 06:12:57.972552 139781364750208 model_lib_v2.py:652] Step 5500 per-step time 0.256s loss=1.085
    INFO:tensorflow:Step 5600 per-step time 0.263s loss=1.180
    I1009 06:13:23.283909 139781364750208 model_lib_v2.py:652] Step 5600 per-step time 0.263s loss=1.180
    INFO:tensorflow:Step 5700 per-step time 0.249s loss=0.825
    I1009 06:13:48.833641 139781364750208 model_lib_v2.py:652] Step 5700 per-step time 0.249s loss=0.825
    INFO:tensorflow:Step 5800 per-step time 0.255s loss=1.011
    I1009 06:14:14.212515 139781364750208 model_lib_v2.py:652] Step 5800 per-step time 0.255s loss=1.011
    INFO:tensorflow:Step 5900 per-step time 0.248s loss=1.131
    I1009 06:14:39.601957 139781364750208 model_lib_v2.py:652] Step 5900 per-step time 0.248s loss=1.131
    INFO:tensorflow:Step 6000 per-step time 0.266s loss=1.157
    I1009 06:15:05.065816 139781364750208 model_lib_v2.py:652] Step 6000 per-step time 0.266s loss=1.157
    INFO:tensorflow:Step 6100 per-step time 0.248s loss=1.259
    I1009 06:15:31.666754 139781364750208 model_lib_v2.py:652] Step 6100 per-step time 0.248s loss=1.259
    INFO:tensorflow:Step 6200 per-step time 0.251s loss=0.944
    I1009 06:15:57.087922 139781364750208 model_lib_v2.py:652] Step 6200 per-step time 0.251s loss=0.944
    INFO:tensorflow:Step 6300 per-step time 0.262s loss=0.964
    I1009 06:16:22.433129 139781364750208 model_lib_v2.py:652] Step 6300 per-step time 0.262s loss=0.964
    INFO:tensorflow:Step 6400 per-step time 0.262s loss=1.164
    I1009 06:16:48.082877 139781364750208 model_lib_v2.py:652] Step 6400 per-step time 0.262s loss=1.164
    INFO:tensorflow:Step 6500 per-step time 0.241s loss=0.973
    I1009 06:17:13.699000 139781364750208 model_lib_v2.py:652] Step 6500 per-step time 0.241s loss=0.973
    INFO:tensorflow:Step 6600 per-step time 0.252s loss=1.390
    I1009 06:17:39.228833 139781364750208 model_lib_v2.py:652] Step 6600 per-step time 0.252s loss=1.390
    INFO:tensorflow:Step 6700 per-step time 0.257s loss=0.935
    I1009 06:18:04.715637 139781364750208 model_lib_v2.py:652] Step 6700 per-step time 0.257s loss=0.935
    INFO:tensorflow:Step 6800 per-step time 0.248s loss=1.244
    I1009 06:18:30.215171 139781364750208 model_lib_v2.py:652] Step 6800 per-step time 0.248s loss=1.244
    INFO:tensorflow:Step 6900 per-step time 0.251s loss=0.812
    I1009 06:18:55.586223 139781364750208 model_lib_v2.py:652] Step 6900 per-step time 0.251s loss=0.812
    INFO:tensorflow:Step 7000 per-step time 0.258s loss=0.732
    I1009 06:19:21.086194 139781364750208 model_lib_v2.py:652] Step 7000 per-step time 0.258s loss=0.732
    INFO:tensorflow:Step 7100 per-step time 0.252s loss=0.859
    I1009 06:19:47.488374 139781364750208 model_lib_v2.py:652] Step 7100 per-step time 0.252s loss=0.859
    INFO:tensorflow:Step 7200 per-step time 0.245s loss=1.253
    I1009 06:20:13.025879 139781364750208 model_lib_v2.py:652] Step 7200 per-step time 0.245s loss=1.253
    INFO:tensorflow:Step 7300 per-step time 0.244s loss=1.071
    I1009 06:20:38.210453 139781364750208 model_lib_v2.py:652] Step 7300 per-step time 0.244s loss=1.071
    INFO:tensorflow:Step 7400 per-step time 0.252s loss=0.865
    I1009 06:21:03.614911 139781364750208 model_lib_v2.py:652] Step 7400 per-step time 0.252s loss=0.865
    INFO:tensorflow:Step 7500 per-step time 0.240s loss=0.854
    I1009 06:21:29.059541 139781364750208 model_lib_v2.py:652] Step 7500 per-step time 0.240s loss=0.854
    INFO:tensorflow:Step 7600 per-step time 0.261s loss=1.061
    I1009 06:21:54.401137 139781364750208 model_lib_v2.py:652] Step 7600 per-step time 0.261s loss=1.061
    INFO:tensorflow:Step 7700 per-step time 0.244s loss=0.903
    I1009 06:22:20.082902 139781364750208 model_lib_v2.py:652] Step 7700 per-step time 0.244s loss=0.903
    INFO:tensorflow:Step 7800 per-step time 0.253s loss=1.275
    I1009 06:22:45.640299 139781364750208 model_lib_v2.py:652] Step 7800 per-step time 0.253s loss=1.275
    INFO:tensorflow:Step 7900 per-step time 0.273s loss=0.878
    I1009 06:23:11.106666 139781364750208 model_lib_v2.py:652] Step 7900 per-step time 0.273s loss=0.878
    INFO:tensorflow:Step 8000 per-step time 0.252s loss=1.309
    I1009 06:23:36.375234 139781364750208 model_lib_v2.py:652] Step 8000 per-step time 0.252s loss=1.309
    INFO:tensorflow:Step 8100 per-step time 0.262s loss=0.686
    I1009 06:24:02.928998 139781364750208 model_lib_v2.py:652] Step 8100 per-step time 0.262s loss=0.686
    INFO:tensorflow:Step 8200 per-step time 0.251s loss=1.138
    I1009 06:24:28.239483 139781364750208 model_lib_v2.py:652] Step 8200 per-step time 0.251s loss=1.138
    INFO:tensorflow:Step 8300 per-step time 0.265s loss=0.900
    I1009 06:24:53.919176 139781364750208 model_lib_v2.py:652] Step 8300 per-step time 0.265s loss=0.900
    INFO:tensorflow:Step 8400 per-step time 0.244s loss=1.112
    I1009 06:25:19.589705 139781364750208 model_lib_v2.py:652] Step 8400 per-step time 0.244s loss=1.112
    INFO:tensorflow:Step 8500 per-step time 0.242s loss=1.083
    I1009 06:25:45.104796 139781364750208 model_lib_v2.py:652] Step 8500 per-step time 0.242s loss=1.083
    INFO:tensorflow:Step 8600 per-step time 0.246s loss=1.094
    I1009 06:26:10.470775 139781364750208 model_lib_v2.py:652] Step 8600 per-step time 0.246s loss=1.094
    INFO:tensorflow:Step 8700 per-step time 0.256s loss=1.052
    I1009 06:26:35.937526 139781364750208 model_lib_v2.py:652] Step 8700 per-step time 0.256s loss=1.052
    INFO:tensorflow:Step 8800 per-step time 0.254s loss=1.036
    I1009 06:27:01.485830 139781364750208 model_lib_v2.py:652] Step 8800 per-step time 0.254s loss=1.036
    INFO:tensorflow:Step 8900 per-step time 0.250s loss=1.200
    I1009 06:27:27.326024 139781364750208 model_lib_v2.py:652] Step 8900 per-step time 0.250s loss=1.200
    INFO:tensorflow:Step 9000 per-step time 0.258s loss=0.746
    I1009 06:27:52.997074 139781364750208 model_lib_v2.py:652] Step 9000 per-step time 0.258s loss=0.746
    INFO:tensorflow:Step 9100 per-step time 0.245s loss=0.779
    I1009 06:28:19.326532 139781364750208 model_lib_v2.py:652] Step 9100 per-step time 0.245s loss=0.779
    INFO:tensorflow:Step 9200 per-step time 0.254s loss=1.230
    I1009 06:28:44.675230 139781364750208 model_lib_v2.py:652] Step 9200 per-step time 0.254s loss=1.230
    INFO:tensorflow:Step 9300 per-step time 0.244s loss=1.310
    I1009 06:29:10.164139 139781364750208 model_lib_v2.py:652] Step 9300 per-step time 0.244s loss=1.310
    INFO:tensorflow:Step 9400 per-step time 0.247s loss=1.186
    I1009 06:29:35.437432 139781364750208 model_lib_v2.py:652] Step 9400 per-step time 0.247s loss=1.186
    INFO:tensorflow:Step 9500 per-step time 0.257s loss=1.023
    I1009 06:30:00.706197 139781364750208 model_lib_v2.py:652] Step 9500 per-step time 0.257s loss=1.023
    INFO:tensorflow:Step 9600 per-step time 0.254s loss=1.284
    I1009 06:30:26.185838 139781364750208 model_lib_v2.py:652] Step 9600 per-step time 0.254s loss=1.284
    INFO:tensorflow:Step 9700 per-step time 0.258s loss=1.309
    I1009 06:30:51.597714 139781364750208 model_lib_v2.py:652] Step 9700 per-step time 0.258s loss=1.309
    INFO:tensorflow:Step 9800 per-step time 0.259s loss=1.087
    I1009 06:31:16.859307 139781364750208 model_lib_v2.py:652] Step 9800 per-step time 0.259s loss=1.087
    INFO:tensorflow:Step 9900 per-step time 0.260s loss=0.875
    I1009 06:31:42.078454 139781364750208 model_lib_v2.py:652] Step 9900 per-step time 0.260s loss=0.875
    INFO:tensorflow:Step 10000 per-step time 0.255s loss=1.178
    I1009 06:32:07.567362 139781364750208 model_lib_v2.py:652] Step 10000 per-step time 0.255s loss=1.178
    INFO:tensorflow:Step 10100 per-step time 0.251s loss=0.946
    I1009 06:32:34.090091 139781364750208 model_lib_v2.py:652] Step 10100 per-step time 0.251s loss=0.946
    INFO:tensorflow:Step 10200 per-step time 0.244s loss=0.966
    I1009 06:32:59.534812 139781364750208 model_lib_v2.py:652] Step 10200 per-step time 0.244s loss=0.966
    INFO:tensorflow:Step 10300 per-step time 0.251s loss=1.268
    I1009 06:33:25.120912 139781364750208 model_lib_v2.py:652] Step 10300 per-step time 0.251s loss=1.268
    INFO:tensorflow:Step 10400 per-step time 0.256s loss=0.830
    I1009 06:33:50.560460 139781364750208 model_lib_v2.py:652] Step 10400 per-step time 0.256s loss=0.830
    INFO:tensorflow:Step 10500 per-step time 0.269s loss=1.193
    I1009 06:34:16.039338 139781364750208 model_lib_v2.py:652] Step 10500 per-step time 0.269s loss=1.193
    INFO:tensorflow:Step 10600 per-step time 0.264s loss=0.710
    I1009 06:34:41.465224 139781364750208 model_lib_v2.py:652] Step 10600 per-step time 0.264s loss=0.710
    INFO:tensorflow:Step 10700 per-step time 0.258s loss=0.877
    I1009 06:35:06.905685 139781364750208 model_lib_v2.py:652] Step 10700 per-step time 0.258s loss=0.877
    INFO:tensorflow:Step 10800 per-step time 0.254s loss=1.126
    I1009 06:35:32.482295 139781364750208 model_lib_v2.py:652] Step 10800 per-step time 0.254s loss=1.126
    INFO:tensorflow:Step 10900 per-step time 0.251s loss=0.552
    I1009 06:35:58.185680 139781364750208 model_lib_v2.py:652] Step 10900 per-step time 0.251s loss=0.552
    INFO:tensorflow:Step 11000 per-step time 0.270s loss=1.136
    I1009 06:36:23.411454 139781364750208 model_lib_v2.py:652] Step 11000 per-step time 0.270s loss=1.136
    INFO:tensorflow:Step 11100 per-step time 0.268s loss=0.804
    I1009 06:36:49.809085 139781364750208 model_lib_v2.py:652] Step 11100 per-step time 0.268s loss=0.804
    INFO:tensorflow:Step 11200 per-step time 0.253s loss=0.736
    I1009 06:37:15.091868 139781364750208 model_lib_v2.py:652] Step 11200 per-step time 0.253s loss=0.736
    INFO:tensorflow:Step 11300 per-step time 0.251s loss=0.942
    I1009 06:37:40.738817 139781364750208 model_lib_v2.py:652] Step 11300 per-step time 0.251s loss=0.942
    INFO:tensorflow:Step 11400 per-step time 0.255s loss=1.409
    I1009 06:38:06.116509 139781364750208 model_lib_v2.py:652] Step 11400 per-step time 0.255s loss=1.409
    INFO:tensorflow:Step 11500 per-step time 0.245s loss=1.022
    I1009 06:38:31.488322 139781364750208 model_lib_v2.py:652] Step 11500 per-step time 0.245s loss=1.022
    INFO:tensorflow:Step 11600 per-step time 0.255s loss=1.195
    I1009 06:38:56.907812 139781364750208 model_lib_v2.py:652] Step 11600 per-step time 0.255s loss=1.195
    INFO:tensorflow:Step 11700 per-step time 0.251s loss=1.172
    I1009 06:39:22.355795 139781364750208 model_lib_v2.py:652] Step 11700 per-step time 0.251s loss=1.172
    INFO:tensorflow:Step 11800 per-step time 0.257s loss=1.373
    I1009 06:39:47.745288 139781364750208 model_lib_v2.py:652] Step 11800 per-step time 0.257s loss=1.373
    INFO:tensorflow:Step 11900 per-step time 0.248s loss=0.843
    I1009 06:40:13.208161 139781364750208 model_lib_v2.py:652] Step 11900 per-step time 0.248s loss=0.843
    INFO:tensorflow:Step 12000 per-step time 0.260s loss=0.820
    I1009 06:40:38.596061 139781364750208 model_lib_v2.py:652] Step 12000 per-step time 0.260s loss=0.820
    INFO:tensorflow:Step 12100 per-step time 0.239s loss=1.029
    I1009 06:41:04.935035 139781364750208 model_lib_v2.py:652] Step 12100 per-step time 0.239s loss=1.029
    INFO:tensorflow:Step 12200 per-step time 0.244s loss=1.304
    I1009 06:41:30.289941 139781364750208 model_lib_v2.py:652] Step 12200 per-step time 0.244s loss=1.304
    INFO:tensorflow:Step 12300 per-step time 0.252s loss=0.961
    I1009 06:41:55.779265 139781364750208 model_lib_v2.py:652] Step 12300 per-step time 0.252s loss=0.961
    INFO:tensorflow:Step 12400 per-step time 0.261s loss=0.776
    I1009 06:42:21.196507 139781364750208 model_lib_v2.py:652] Step 12400 per-step time 0.261s loss=0.776
    INFO:tensorflow:Step 12500 per-step time 0.269s loss=1.033
    I1009 06:42:46.533617 139781364750208 model_lib_v2.py:652] Step 12500 per-step time 0.269s loss=1.033
    INFO:tensorflow:Step 12600 per-step time 0.256s loss=0.950
    I1009 06:43:12.143428 139781364750208 model_lib_v2.py:652] Step 12600 per-step time 0.256s loss=0.950
    INFO:tensorflow:Step 12700 per-step time 0.250s loss=1.052
    I1009 06:43:37.577159 139781364750208 model_lib_v2.py:652] Step 12700 per-step time 0.250s loss=1.052
    INFO:tensorflow:Step 12800 per-step time 0.248s loss=0.870
    I1009 06:44:02.896934 139781364750208 model_lib_v2.py:652] Step 12800 per-step time 0.248s loss=0.870
    INFO:tensorflow:Step 12900 per-step time 0.266s loss=1.110
    I1009 06:44:28.168969 139781364750208 model_lib_v2.py:652] Step 12900 per-step time 0.266s loss=1.110
    INFO:tensorflow:Step 13000 per-step time 0.258s loss=1.670
    I1009 06:44:53.464618 139781364750208 model_lib_v2.py:652] Step 13000 per-step time 0.258s loss=1.670
    INFO:tensorflow:Step 13100 per-step time 0.261s loss=1.175
    I1009 06:45:19.807616 139781364750208 model_lib_v2.py:652] Step 13100 per-step time 0.261s loss=1.175
    INFO:tensorflow:Step 13200 per-step time 0.252s loss=0.756
    I1009 06:45:45.461234 139781364750208 model_lib_v2.py:652] Step 13200 per-step time 0.252s loss=0.756
    INFO:tensorflow:Step 13300 per-step time 0.251s loss=0.957
    I1009 06:46:10.783417 139781364750208 model_lib_v2.py:652] Step 13300 per-step time 0.251s loss=0.957
    INFO:tensorflow:Step 13400 per-step time 0.250s loss=0.997
    I1009 06:46:36.324150 139781364750208 model_lib_v2.py:652] Step 13400 per-step time 0.250s loss=0.997
    INFO:tensorflow:Step 13500 per-step time 0.260s loss=0.808
    I1009 06:47:01.817976 139781364750208 model_lib_v2.py:652] Step 13500 per-step time 0.260s loss=0.808
    INFO:tensorflow:Step 13600 per-step time 0.247s loss=0.810
    I1009 06:47:27.259854 139781364750208 model_lib_v2.py:652] Step 13600 per-step time 0.247s loss=0.810
    INFO:tensorflow:Step 13700 per-step time 0.261s loss=0.957
    I1009 06:47:52.667975 139781364750208 model_lib_v2.py:652] Step 13700 per-step time 0.261s loss=0.957
    INFO:tensorflow:Step 13800 per-step time 0.258s loss=0.977
    I1009 06:48:18.290258 139781364750208 model_lib_v2.py:652] Step 13800 per-step time 0.258s loss=0.977
    INFO:tensorflow:Step 13900 per-step time 0.253s loss=0.915
    I1009 06:48:43.772617 139781364750208 model_lib_v2.py:652] Step 13900 per-step time 0.253s loss=0.915
    INFO:tensorflow:Step 14000 per-step time 0.248s loss=0.775
    I1009 06:49:09.298063 139781364750208 model_lib_v2.py:652] Step 14000 per-step time 0.248s loss=0.775
    INFO:tensorflow:Step 14100 per-step time 0.250s loss=0.954
    I1009 06:49:35.908318 139781364750208 model_lib_v2.py:652] Step 14100 per-step time 0.250s loss=0.954
    INFO:tensorflow:Step 14200 per-step time 0.251s loss=1.250
    I1009 06:50:01.191612 139781364750208 model_lib_v2.py:652] Step 14200 per-step time 0.251s loss=1.250
    INFO:tensorflow:Step 14300 per-step time 0.263s loss=1.404
    I1009 06:50:26.820368 139781364750208 model_lib_v2.py:652] Step 14300 per-step time 0.263s loss=1.404
    INFO:tensorflow:Step 14400 per-step time 0.252s loss=1.116
    I1009 06:50:52.485661 139781364750208 model_lib_v2.py:652] Step 14400 per-step time 0.252s loss=1.116
    INFO:tensorflow:Step 14500 per-step time 0.252s loss=1.070
    I1009 06:51:17.919413 139781364750208 model_lib_v2.py:652] Step 14500 per-step time 0.252s loss=1.070
    INFO:tensorflow:Step 14600 per-step time 0.252s loss=1.269
    I1009 06:51:43.253190 139781364750208 model_lib_v2.py:652] Step 14600 per-step time 0.252s loss=1.269
    INFO:tensorflow:Step 14700 per-step time 0.251s loss=1.215
    I1009 06:52:08.762670 139781364750208 model_lib_v2.py:652] Step 14700 per-step time 0.251s loss=1.215
    INFO:tensorflow:Step 14800 per-step time 0.258s loss=0.898
    I1009 06:52:34.300161 139781364750208 model_lib_v2.py:652] Step 14800 per-step time 0.258s loss=0.898
    INFO:tensorflow:Step 14900 per-step time 0.262s loss=0.734
    I1009 06:52:59.991937 139781364750208 model_lib_v2.py:652] Step 14900 per-step time 0.262s loss=0.734
    INFO:tensorflow:Step 15000 per-step time 0.242s loss=1.376
    I1009 06:53:25.687745 139781364750208 model_lib_v2.py:652] Step 15000 per-step time 0.242s loss=1.376
    INFO:tensorflow:Step 15100 per-step time 0.253s loss=1.001
    I1009 06:53:52.243430 139781364750208 model_lib_v2.py:652] Step 15100 per-step time 0.253s loss=1.001
    INFO:tensorflow:Step 15200 per-step time 0.262s loss=0.647
    I1009 06:54:17.849664 139781364750208 model_lib_v2.py:652] Step 15200 per-step time 0.262s loss=0.647
    INFO:tensorflow:Step 15300 per-step time 0.271s loss=0.824
    I1009 06:54:43.554188 139781364750208 model_lib_v2.py:652] Step 15300 per-step time 0.271s loss=0.824
    INFO:tensorflow:Step 15400 per-step time 0.265s loss=1.277
    I1009 06:55:09.088019 139781364750208 model_lib_v2.py:652] Step 15400 per-step time 0.265s loss=1.277
    INFO:tensorflow:Step 15500 per-step time 0.262s loss=0.893
    I1009 06:55:34.655333 139781364750208 model_lib_v2.py:652] Step 15500 per-step time 0.262s loss=0.893
    INFO:tensorflow:Step 15600 per-step time 0.257s loss=0.634
    I1009 06:56:00.164465 139781364750208 model_lib_v2.py:652] Step 15600 per-step time 0.257s loss=0.634
    INFO:tensorflow:Step 15700 per-step time 0.252s loss=0.674
    I1009 06:56:25.575020 139781364750208 model_lib_v2.py:652] Step 15700 per-step time 0.252s loss=0.674
    INFO:tensorflow:Step 15800 per-step time 0.258s loss=1.137
    I1009 06:56:50.972581 139781364750208 model_lib_v2.py:652] Step 15800 per-step time 0.258s loss=1.137
    INFO:tensorflow:Step 15900 per-step time 0.260s loss=0.730
    I1009 06:57:16.460677 139781364750208 model_lib_v2.py:652] Step 15900 per-step time 0.260s loss=0.730
    INFO:tensorflow:Step 16000 per-step time 0.263s loss=0.881
    I1009 06:57:41.896816 139781364750208 model_lib_v2.py:652] Step 16000 per-step time 0.263s loss=0.881
    INFO:tensorflow:Step 16100 per-step time 0.256s loss=0.830
    I1009 06:58:08.423965 139781364750208 model_lib_v2.py:652] Step 16100 per-step time 0.256s loss=0.830
    INFO:tensorflow:Step 16200 per-step time 0.253s loss=1.487
    I1009 06:58:34.153488 139781364750208 model_lib_v2.py:652] Step 16200 per-step time 0.253s loss=1.487
    INFO:tensorflow:Step 16300 per-step time 0.244s loss=1.068
    I1009 06:58:59.759288 139781364750208 model_lib_v2.py:652] Step 16300 per-step time 0.244s loss=1.068
    INFO:tensorflow:Step 16400 per-step time 0.256s loss=1.126
    I1009 06:59:25.491960 139781364750208 model_lib_v2.py:652] Step 16400 per-step time 0.256s loss=1.126
    INFO:tensorflow:Step 16500 per-step time 0.253s loss=0.996
    I1009 06:59:50.845860 139781364750208 model_lib_v2.py:652] Step 16500 per-step time 0.253s loss=0.996
    INFO:tensorflow:Step 16600 per-step time 0.260s loss=0.823
    I1009 07:00:16.344214 139781364750208 model_lib_v2.py:652] Step 16600 per-step time 0.260s loss=0.823
    INFO:tensorflow:Step 16700 per-step time 0.243s loss=0.883
    I1009 07:00:41.852163 139781364750208 model_lib_v2.py:652] Step 16700 per-step time 0.243s loss=0.883
    INFO:tensorflow:Step 16800 per-step time 0.242s loss=1.190
    I1009 07:01:07.268170 139781364750208 model_lib_v2.py:652] Step 16800 per-step time 0.242s loss=1.190
    INFO:tensorflow:Step 16900 per-step time 0.248s loss=1.104
    I1009 07:01:32.735877 139781364750208 model_lib_v2.py:652] Step 16900 per-step time 0.248s loss=1.104
    INFO:tensorflow:Step 17000 per-step time 0.258s loss=1.089
    I1009 07:01:58.316429 139781364750208 model_lib_v2.py:652] Step 17000 per-step time 0.258s loss=1.089
    INFO:tensorflow:Step 17100 per-step time 0.249s loss=1.443
    I1009 07:02:24.711697 139781364750208 model_lib_v2.py:652] Step 17100 per-step time 0.249s loss=1.443
    INFO:tensorflow:Step 17200 per-step time 0.256s loss=1.292
    I1009 07:02:50.094846 139781364750208 model_lib_v2.py:652] Step 17200 per-step time 0.256s loss=1.292
    INFO:tensorflow:Step 17300 per-step time 0.262s loss=1.354
    I1009 07:03:15.499893 139781364750208 model_lib_v2.py:652] Step 17300 per-step time 0.262s loss=1.354
    INFO:tensorflow:Step 17400 per-step time 0.253s loss=0.849
    I1009 07:03:41.169132 139781364750208 model_lib_v2.py:652] Step 17400 per-step time 0.253s loss=0.849
    INFO:tensorflow:Step 17500 per-step time 0.249s loss=1.126
    I1009 07:04:06.514449 139781364750208 model_lib_v2.py:652] Step 17500 per-step time 0.249s loss=1.126
    INFO:tensorflow:Step 17600 per-step time 0.255s loss=0.955
    I1009 07:04:31.980645 139781364750208 model_lib_v2.py:652] Step 17600 per-step time 0.255s loss=0.955
    INFO:tensorflow:Step 17700 per-step time 0.252s loss=1.158
    I1009 07:04:57.461558 139781364750208 model_lib_v2.py:652] Step 17700 per-step time 0.252s loss=1.158
    INFO:tensorflow:Step 17800 per-step time 0.273s loss=0.951
    I1009 07:05:22.809362 139781364750208 model_lib_v2.py:652] Step 17800 per-step time 0.273s loss=0.951
    INFO:tensorflow:Step 17900 per-step time 0.241s loss=0.929
    I1009 07:05:48.216137 139781364750208 model_lib_v2.py:652] Step 17900 per-step time 0.241s loss=0.929
    INFO:tensorflow:Step 18000 per-step time 0.252s loss=1.153
    I1009 07:06:13.664760 139781364750208 model_lib_v2.py:652] Step 18000 per-step time 0.252s loss=1.153
    INFO:tensorflow:Step 18100 per-step time 0.260s loss=0.570
    I1009 07:06:40.159097 139781364750208 model_lib_v2.py:652] Step 18100 per-step time 0.260s loss=0.570
    INFO:tensorflow:Step 18200 per-step time 0.268s loss=1.353
    I1009 07:07:05.539010 139781364750208 model_lib_v2.py:652] Step 18200 per-step time 0.268s loss=1.353
    INFO:tensorflow:Step 18300 per-step time 0.244s loss=1.035
    I1009 07:07:30.997150 139781364750208 model_lib_v2.py:652] Step 18300 per-step time 0.244s loss=1.035
    INFO:tensorflow:Step 18400 per-step time 0.242s loss=0.921
    I1009 07:07:56.507236 139781364750208 model_lib_v2.py:652] Step 18400 per-step time 0.242s loss=0.921
    INFO:tensorflow:Step 18500 per-step time 0.261s loss=1.495
    I1009 07:08:21.948672 139781364750208 model_lib_v2.py:652] Step 18500 per-step time 0.261s loss=1.495
    INFO:tensorflow:Step 18600 per-step time 0.248s loss=0.890
    I1009 07:08:47.590320 139781364750208 model_lib_v2.py:652] Step 18600 per-step time 0.248s loss=0.890
    INFO:tensorflow:Step 18700 per-step time 0.260s loss=0.969
    I1009 07:09:13.014168 139781364750208 model_lib_v2.py:652] Step 18700 per-step time 0.260s loss=0.969
    INFO:tensorflow:Step 18800 per-step time 0.261s loss=1.300
    I1009 07:09:38.577408 139781364750208 model_lib_v2.py:652] Step 18800 per-step time 0.261s loss=1.300
    INFO:tensorflow:Step 18900 per-step time 0.247s loss=1.157
    I1009 07:10:03.866135 139781364750208 model_lib_v2.py:652] Step 18900 per-step time 0.247s loss=1.157
    INFO:tensorflow:Step 19000 per-step time 0.257s loss=0.772
    I1009 07:10:29.131218 139781364750208 model_lib_v2.py:652] Step 19000 per-step time 0.257s loss=0.772
    INFO:tensorflow:Step 19100 per-step time 0.277s loss=1.139
    I1009 07:10:55.538635 139781364750208 model_lib_v2.py:652] Step 19100 per-step time 0.277s loss=1.139
    INFO:tensorflow:Step 19200 per-step time 0.276s loss=0.781
    I1009 07:11:21.022017 139781364750208 model_lib_v2.py:652] Step 19200 per-step time 0.276s loss=0.781
    INFO:tensorflow:Step 19300 per-step time 0.247s loss=0.765
    I1009 07:11:46.417162 139781364750208 model_lib_v2.py:652] Step 19300 per-step time 0.247s loss=0.765
    INFO:tensorflow:Step 19400 per-step time 0.257s loss=1.115
    I1009 07:12:11.896707 139781364750208 model_lib_v2.py:652] Step 19400 per-step time 0.257s loss=1.115
    INFO:tensorflow:Step 19500 per-step time 0.258s loss=1.140
    I1009 07:12:37.442597 139781364750208 model_lib_v2.py:652] Step 19500 per-step time 0.258s loss=1.140
    INFO:tensorflow:Step 19600 per-step time 0.249s loss=0.900
    I1009 07:13:02.894373 139781364750208 model_lib_v2.py:652] Step 19600 per-step time 0.249s loss=0.900
    INFO:tensorflow:Step 19700 per-step time 0.252s loss=0.935
    I1009 07:13:28.267908 139781364750208 model_lib_v2.py:652] Step 19700 per-step time 0.252s loss=0.935
    INFO:tensorflow:Step 19800 per-step time 0.265s loss=1.034
    I1009 07:13:53.838116 139781364750208 model_lib_v2.py:652] Step 19800 per-step time 0.265s loss=1.034
    INFO:tensorflow:Step 19900 per-step time 0.253s loss=0.991
    I1009 07:14:19.184275 139781364750208 model_lib_v2.py:652] Step 19900 per-step time 0.253s loss=0.991
    INFO:tensorflow:Step 20000 per-step time 0.244s loss=0.918
    I1009 07:14:44.497664 139781364750208 model_lib_v2.py:652] Step 20000 per-step time 0.244s loss=0.918
    INFO:tensorflow:Step 20100 per-step time 0.260s loss=0.995
    I1009 07:15:10.601821 139781364750208 model_lib_v2.py:652] Step 20100 per-step time 0.260s loss=0.995
    INFO:tensorflow:Step 20200 per-step time 0.248s loss=1.077
    I1009 07:15:35.748328 139781364750208 model_lib_v2.py:652] Step 20200 per-step time 0.248s loss=1.077
    INFO:tensorflow:Step 20300 per-step time 0.260s loss=0.909
    I1009 07:16:01.109660 139781364750208 model_lib_v2.py:652] Step 20300 per-step time 0.260s loss=0.909
    INFO:tensorflow:Step 20400 per-step time 0.251s loss=0.782
    I1009 07:16:26.610617 139781364750208 model_lib_v2.py:652] Step 20400 per-step time 0.251s loss=0.782
    INFO:tensorflow:Step 20500 per-step time 0.246s loss=1.247
    I1009 07:16:51.945923 139781364750208 model_lib_v2.py:652] Step 20500 per-step time 0.246s loss=1.247
    INFO:tensorflow:Step 20600 per-step time 0.254s loss=1.216
    I1009 07:17:17.254567 139781364750208 model_lib_v2.py:652] Step 20600 per-step time 0.254s loss=1.216
    INFO:tensorflow:Step 20700 per-step time 0.254s loss=0.732
    I1009 07:17:42.532540 139781364750208 model_lib_v2.py:652] Step 20700 per-step time 0.254s loss=0.732
    INFO:tensorflow:Step 20800 per-step time 0.255s loss=1.101
    I1009 07:18:07.755675 139781364750208 model_lib_v2.py:652] Step 20800 per-step time 0.255s loss=1.101
    INFO:tensorflow:Step 20900 per-step time 0.253s loss=1.144
    I1009 07:18:32.910790 139781364750208 model_lib_v2.py:652] Step 20900 per-step time 0.253s loss=1.144
    INFO:tensorflow:Step 21000 per-step time 0.259s loss=0.674
    I1009 07:18:58.188032 139781364750208 model_lib_v2.py:652] Step 21000 per-step time 0.259s loss=0.674
    INFO:tensorflow:Step 21100 per-step time 0.251s loss=1.021
    I1009 07:19:25.432892 139781364750208 model_lib_v2.py:652] Step 21100 per-step time 0.251s loss=1.021
    INFO:tensorflow:Step 21200 per-step time 0.255s loss=0.894
    I1009 07:19:50.863998 139781364750208 model_lib_v2.py:652] Step 21200 per-step time 0.255s loss=0.894
    INFO:tensorflow:Step 21300 per-step time 0.250s loss=0.873
    I1009 07:20:16.166074 139781364750208 model_lib_v2.py:652] Step 21300 per-step time 0.250s loss=0.873
    INFO:tensorflow:Step 21400 per-step time 0.246s loss=0.973
    I1009 07:20:41.330326 139781364750208 model_lib_v2.py:652] Step 21400 per-step time 0.246s loss=0.973
    INFO:tensorflow:Step 21500 per-step time 0.259s loss=1.084
    I1009 07:21:06.947378 139781364750208 model_lib_v2.py:652] Step 21500 per-step time 0.259s loss=1.084
    INFO:tensorflow:Step 21600 per-step time 0.246s loss=1.205
    I1009 07:21:32.293313 139781364750208 model_lib_v2.py:652] Step 21600 per-step time 0.246s loss=1.205
    INFO:tensorflow:Step 21700 per-step time 0.242s loss=0.875
    I1009 07:21:57.439709 139781364750208 model_lib_v2.py:652] Step 21700 per-step time 0.242s loss=0.875
    INFO:tensorflow:Step 21800 per-step time 0.244s loss=0.866
    I1009 07:22:22.757441 139781364750208 model_lib_v2.py:652] Step 21800 per-step time 0.244s loss=0.866
    INFO:tensorflow:Step 21900 per-step time 0.258s loss=0.856
    I1009 07:22:47.771777 139781364750208 model_lib_v2.py:652] Step 21900 per-step time 0.258s loss=0.856
    INFO:tensorflow:Step 22000 per-step time 0.255s loss=1.795
    I1009 07:23:12.991371 139781364750208 model_lib_v2.py:652] Step 22000 per-step time 0.255s loss=1.795
    INFO:tensorflow:Step 22100 per-step time 0.246s loss=0.986
    I1009 07:23:39.044558 139781364750208 model_lib_v2.py:652] Step 22100 per-step time 0.246s loss=0.986
    INFO:tensorflow:Step 22200 per-step time 0.250s loss=1.109
    I1009 07:24:04.380810 139781364750208 model_lib_v2.py:652] Step 22200 per-step time 0.250s loss=1.109
    INFO:tensorflow:Step 22300 per-step time 0.247s loss=0.882
    I1009 07:24:30.089944 139781364750208 model_lib_v2.py:652] Step 22300 per-step time 0.247s loss=0.882
    INFO:tensorflow:Step 22400 per-step time 0.252s loss=0.806
    I1009 07:24:55.512823 139781364750208 model_lib_v2.py:652] Step 22400 per-step time 0.252s loss=0.806
    INFO:tensorflow:Step 22500 per-step time 0.240s loss=0.907
    I1009 07:25:20.887675 139781364750208 model_lib_v2.py:652] Step 22500 per-step time 0.240s loss=0.907
    INFO:tensorflow:Step 22600 per-step time 0.269s loss=0.883
    I1009 07:25:45.975736 139781364750208 model_lib_v2.py:652] Step 22600 per-step time 0.269s loss=0.883
    INFO:tensorflow:Step 22700 per-step time 0.255s loss=0.811
    I1009 07:26:11.274186 139781364750208 model_lib_v2.py:652] Step 22700 per-step time 0.255s loss=0.811
    INFO:tensorflow:Step 22800 per-step time 0.257s loss=1.259
    I1009 07:26:36.261443 139781364750208 model_lib_v2.py:652] Step 22800 per-step time 0.257s loss=1.259
    INFO:tensorflow:Step 22900 per-step time 0.265s loss=1.081
    I1009 07:27:01.403154 139781364750208 model_lib_v2.py:652] Step 22900 per-step time 0.265s loss=1.081
    INFO:tensorflow:Step 23000 per-step time 0.241s loss=0.821
    I1009 07:27:26.360065 139781364750208 model_lib_v2.py:652] Step 23000 per-step time 0.241s loss=0.821
    INFO:tensorflow:Step 23100 per-step time 0.246s loss=0.763
    I1009 07:27:52.427942 139781364750208 model_lib_v2.py:652] Step 23100 per-step time 0.246s loss=0.763
    INFO:tensorflow:Step 23200 per-step time 0.270s loss=0.990
    I1009 07:28:17.656212 139781364750208 model_lib_v2.py:652] Step 23200 per-step time 0.270s loss=0.990
    INFO:tensorflow:Step 23300 per-step time 0.256s loss=1.179
    I1009 07:28:42.903771 139781364750208 model_lib_v2.py:652] Step 23300 per-step time 0.256s loss=1.179
    INFO:tensorflow:Step 23400 per-step time 0.249s loss=1.231
    I1009 07:29:08.166598 139781364750208 model_lib_v2.py:652] Step 23400 per-step time 0.249s loss=1.231
    INFO:tensorflow:Step 23500 per-step time 0.252s loss=1.176
    I1009 07:29:33.637335 139781364750208 model_lib_v2.py:652] Step 23500 per-step time 0.252s loss=1.176
    INFO:tensorflow:Step 23600 per-step time 0.257s loss=1.126
    I1009 07:29:58.919677 139781364750208 model_lib_v2.py:652] Step 23600 per-step time 0.257s loss=1.126
    INFO:tensorflow:Step 23700 per-step time 0.260s loss=1.125
    I1009 07:30:24.204113 139781364750208 model_lib_v2.py:652] Step 23700 per-step time 0.260s loss=1.125
    INFO:tensorflow:Step 23800 per-step time 0.252s loss=1.095
    I1009 07:30:49.425261 139781364750208 model_lib_v2.py:652] Step 23800 per-step time 0.252s loss=1.095
    INFO:tensorflow:Step 23900 per-step time 0.251s loss=1.057
    I1009 07:31:14.732045 139781364750208 model_lib_v2.py:652] Step 23900 per-step time 0.251s loss=1.057
    INFO:tensorflow:Step 24000 per-step time 0.257s loss=0.991
    I1009 07:31:39.850579 139781364750208 model_lib_v2.py:652] Step 24000 per-step time 0.257s loss=0.991
    INFO:tensorflow:Step 24100 per-step time 0.263s loss=1.019
    I1009 07:32:06.020779 139781364750208 model_lib_v2.py:652] Step 24100 per-step time 0.263s loss=1.019
    INFO:tensorflow:Step 24200 per-step time 0.269s loss=0.723
    I1009 07:32:31.339191 139781364750208 model_lib_v2.py:652] Step 24200 per-step time 0.269s loss=0.723
    INFO:tensorflow:Step 24300 per-step time 0.255s loss=1.237
    I1009 07:32:56.614558 139781364750208 model_lib_v2.py:652] Step 24300 per-step time 0.255s loss=1.237
    INFO:tensorflow:Step 24400 per-step time 0.255s loss=1.194
    I1009 07:33:21.671629 139781364750208 model_lib_v2.py:652] Step 24400 per-step time 0.255s loss=1.194
    INFO:tensorflow:Step 24500 per-step time 0.259s loss=1.266
    I1009 07:33:46.953236 139781364750208 model_lib_v2.py:652] Step 24500 per-step time 0.259s loss=1.266
    INFO:tensorflow:Step 24600 per-step time 0.258s loss=0.923
    I1009 07:34:11.921406 139781364750208 model_lib_v2.py:652] Step 24600 per-step time 0.258s loss=0.923
    INFO:tensorflow:Step 24700 per-step time 0.241s loss=1.088
    I1009 07:34:37.106459 139781364750208 model_lib_v2.py:652] Step 24700 per-step time 0.241s loss=1.088
    INFO:tensorflow:Step 24800 per-step time 0.251s loss=0.727
    I1009 07:35:02.115993 139781364750208 model_lib_v2.py:652] Step 24800 per-step time 0.251s loss=0.727
    INFO:tensorflow:Step 24900 per-step time 0.255s loss=0.645
    I1009 07:35:27.082267 139781364750208 model_lib_v2.py:652] Step 24900 per-step time 0.255s loss=0.645
    INFO:tensorflow:Step 25000 per-step time 0.238s loss=0.927
    I1009 07:35:51.969834 139781364750208 model_lib_v2.py:652] Step 25000 per-step time 0.238s loss=0.927
    INFO:tensorflow:Step 25100 per-step time 0.265s loss=1.098
    I1009 07:36:18.114956 139781364750208 model_lib_v2.py:652] Step 25100 per-step time 0.265s loss=1.098
    INFO:tensorflow:Step 25200 per-step time 0.238s loss=0.857
    I1009 07:36:42.817028 139781364750208 model_lib_v2.py:652] Step 25200 per-step time 0.238s loss=0.857
    INFO:tensorflow:Step 25300 per-step time 0.240s loss=1.239
    I1009 07:37:07.711950 139781364750208 model_lib_v2.py:652] Step 25300 per-step time 0.240s loss=1.239
    INFO:tensorflow:Step 25400 per-step time 0.248s loss=1.049
    I1009 07:37:32.606049 139781364750208 model_lib_v2.py:652] Step 25400 per-step time 0.248s loss=1.049
    INFO:tensorflow:Step 25500 per-step time 0.248s loss=1.201
    I1009 07:37:57.439289 139781364750208 model_lib_v2.py:652] Step 25500 per-step time 0.248s loss=1.201
    INFO:tensorflow:Step 25600 per-step time 0.245s loss=0.985
    I1009 07:38:22.336433 139781364750208 model_lib_v2.py:652] Step 25600 per-step time 0.245s loss=0.985
    INFO:tensorflow:Step 25700 per-step time 0.233s loss=1.545
    I1009 07:38:47.071678 139781364750208 model_lib_v2.py:652] Step 25700 per-step time 0.233s loss=1.545
    INFO:tensorflow:Step 25800 per-step time 0.238s loss=0.809
    I1009 07:39:11.828410 139781364750208 model_lib_v2.py:652] Step 25800 per-step time 0.238s loss=0.809
    INFO:tensorflow:Step 25900 per-step time 0.244s loss=0.839
    I1009 07:39:36.704581 139781364750208 model_lib_v2.py:652] Step 25900 per-step time 0.244s loss=0.839
    INFO:tensorflow:Step 26000 per-step time 0.257s loss=0.918
    I1009 07:40:01.677390 139781364750208 model_lib_v2.py:652] Step 26000 per-step time 0.257s loss=0.918
    INFO:tensorflow:Step 26100 per-step time 0.263s loss=1.262
    I1009 07:40:27.424250 139781364750208 model_lib_v2.py:652] Step 26100 per-step time 0.263s loss=1.262
    INFO:tensorflow:Step 26200 per-step time 0.262s loss=1.124
    I1009 07:40:52.220381 139781364750208 model_lib_v2.py:652] Step 26200 per-step time 0.262s loss=1.124
    INFO:tensorflow:Step 26300 per-step time 0.237s loss=1.026
    I1009 07:41:16.859722 139781364750208 model_lib_v2.py:652] Step 26300 per-step time 0.237s loss=1.026
    INFO:tensorflow:Step 26400 per-step time 0.254s loss=1.199
    I1009 07:41:41.472938 139781364750208 model_lib_v2.py:652] Step 26400 per-step time 0.254s loss=1.199
    INFO:tensorflow:Step 26500 per-step time 0.239s loss=0.856
    I1009 07:42:06.162965 139781364750208 model_lib_v2.py:652] Step 26500 per-step time 0.239s loss=0.856
    INFO:tensorflow:Step 26600 per-step time 0.246s loss=1.102
    I1009 07:42:30.849197 139781364750208 model_lib_v2.py:652] Step 26600 per-step time 0.246s loss=1.102
    INFO:tensorflow:Step 26700 per-step time 0.246s loss=0.947
    I1009 07:42:55.593663 139781364750208 model_lib_v2.py:652] Step 26700 per-step time 0.246s loss=0.947
    INFO:tensorflow:Step 26800 per-step time 0.245s loss=0.921
    I1009 07:43:20.117020 139781364750208 model_lib_v2.py:652] Step 26800 per-step time 0.245s loss=0.921
    INFO:tensorflow:Step 26900 per-step time 0.249s loss=1.345
    I1009 07:43:44.674847 139781364750208 model_lib_v2.py:652] Step 26900 per-step time 0.249s loss=1.345
    INFO:tensorflow:Step 27000 per-step time 0.235s loss=0.847
    I1009 07:44:09.074427 139781364750208 model_lib_v2.py:652] Step 27000 per-step time 0.235s loss=0.847
    INFO:tensorflow:Step 27100 per-step time 0.249s loss=1.145
    I1009 07:44:34.779224 139781364750208 model_lib_v2.py:652] Step 27100 per-step time 0.249s loss=1.145
    INFO:tensorflow:Step 27200 per-step time 0.244s loss=1.107
    I1009 07:44:59.412705 139781364750208 model_lib_v2.py:652] Step 27200 per-step time 0.244s loss=1.107
    INFO:tensorflow:Step 27300 per-step time 0.234s loss=0.523
    I1009 07:45:24.093541 139781364750208 model_lib_v2.py:652] Step 27300 per-step time 0.234s loss=0.523
    INFO:tensorflow:Step 27400 per-step time 0.258s loss=0.913
    I1009 07:45:48.496940 139781364750208 model_lib_v2.py:652] Step 27400 per-step time 0.258s loss=0.913
    INFO:tensorflow:Step 27500 per-step time 0.245s loss=0.817
    I1009 07:46:12.807504 139781364750208 model_lib_v2.py:652] Step 27500 per-step time 0.245s loss=0.817
    INFO:tensorflow:Step 27600 per-step time 0.249s loss=0.843
    I1009 07:46:37.608394 139781364750208 model_lib_v2.py:652] Step 27600 per-step time 0.249s loss=0.843
    INFO:tensorflow:Step 27700 per-step time 0.238s loss=0.867
    I1009 07:47:02.053984 139781364750208 model_lib_v2.py:652] Step 27700 per-step time 0.238s loss=0.867
    INFO:tensorflow:Step 27800 per-step time 0.246s loss=1.195
    I1009 07:47:26.487312 139781364750208 model_lib_v2.py:652] Step 27800 per-step time 0.246s loss=1.195
    INFO:tensorflow:Step 27900 per-step time 0.253s loss=0.649
    I1009 07:47:50.857035 139781364750208 model_lib_v2.py:652] Step 27900 per-step time 0.253s loss=0.649
    INFO:tensorflow:Step 28000 per-step time 0.239s loss=1.153
    I1009 07:48:15.192652 139781364750208 model_lib_v2.py:652] Step 28000 per-step time 0.239s loss=1.153
    INFO:tensorflow:Step 28100 per-step time 0.245s loss=0.865
    I1009 07:48:40.182691 139781364750208 model_lib_v2.py:652] Step 28100 per-step time 0.245s loss=0.865
    INFO:tensorflow:Step 28200 per-step time 0.235s loss=0.912
    I1009 07:49:04.515049 139781364750208 model_lib_v2.py:652] Step 28200 per-step time 0.235s loss=0.912
    INFO:tensorflow:Step 28300 per-step time 0.232s loss=1.114
    I1009 07:49:28.820775 139781364750208 model_lib_v2.py:652] Step 28300 per-step time 0.232s loss=1.114
    INFO:tensorflow:Step 28400 per-step time 0.250s loss=1.017
    I1009 07:49:52.950424 139781364750208 model_lib_v2.py:652] Step 28400 per-step time 0.250s loss=1.017
    INFO:tensorflow:Step 28500 per-step time 0.237s loss=1.097
    I1009 07:50:17.612874 139781364750208 model_lib_v2.py:652] Step 28500 per-step time 0.237s loss=1.097
    INFO:tensorflow:Step 28600 per-step time 0.242s loss=0.681
    I1009 07:50:42.120015 139781364750208 model_lib_v2.py:652] Step 28600 per-step time 0.242s loss=0.681
    INFO:tensorflow:Step 28700 per-step time 0.241s loss=0.650
    I1009 07:51:06.683059 139781364750208 model_lib_v2.py:652] Step 28700 per-step time 0.241s loss=0.650
    INFO:tensorflow:Step 28800 per-step time 0.235s loss=0.880
    I1009 07:51:31.356744 139781364750208 model_lib_v2.py:652] Step 28800 per-step time 0.235s loss=0.880
    INFO:tensorflow:Step 28900 per-step time 0.241s loss=0.931
    I1009 07:51:56.094048 139781364750208 model_lib_v2.py:652] Step 28900 per-step time 0.241s loss=0.931
    INFO:tensorflow:Step 29000 per-step time 0.247s loss=0.928
    I1009 07:52:20.981597 139781364750208 model_lib_v2.py:652] Step 29000 per-step time 0.247s loss=0.928
    INFO:tensorflow:Step 29100 per-step time 0.247s loss=0.708
    I1009 07:52:46.953437 139781364750208 model_lib_v2.py:652] Step 29100 per-step time 0.247s loss=0.708
    INFO:tensorflow:Step 29200 per-step time 0.249s loss=0.975
    I1009 07:53:11.792833 139781364750208 model_lib_v2.py:652] Step 29200 per-step time 0.249s loss=0.975
    INFO:tensorflow:Step 29300 per-step time 0.261s loss=0.905
    I1009 07:53:36.916065 139781364750208 model_lib_v2.py:652] Step 29300 per-step time 0.261s loss=0.905
    INFO:tensorflow:Step 29400 per-step time 0.241s loss=1.208
    I1009 07:54:01.870186 139781364750208 model_lib_v2.py:652] Step 29400 per-step time 0.241s loss=1.208
    INFO:tensorflow:Step 29500 per-step time 0.250s loss=0.644
    I1009 07:54:26.793414 139781364750208 model_lib_v2.py:652] Step 29500 per-step time 0.250s loss=0.644
    INFO:tensorflow:Step 29600 per-step time 0.261s loss=0.710
    I1009 07:54:51.783551 139781364750208 model_lib_v2.py:652] Step 29600 per-step time 0.261s loss=0.710
    INFO:tensorflow:Step 29700 per-step time 0.269s loss=0.888
    I1009 07:55:16.854437 139781364750208 model_lib_v2.py:652] Step 29700 per-step time 0.269s loss=0.888
    INFO:tensorflow:Step 29800 per-step time 0.252s loss=1.607
    I1009 07:55:42.086160 139781364750208 model_lib_v2.py:652] Step 29800 per-step time 0.252s loss=1.607
    INFO:tensorflow:Step 29900 per-step time 0.252s loss=0.703
    I1009 07:56:07.147796 139781364750208 model_lib_v2.py:652] Step 29900 per-step time 0.252s loss=0.703
    INFO:tensorflow:Step 30000 per-step time 0.249s loss=1.075
    I1009 07:56:32.065730 139781364750208 model_lib_v2.py:652] Step 30000 per-step time 0.249s loss=1.075
    INFO:tensorflow:Step 30100 per-step time 0.263s loss=1.367
    I1009 07:56:57.952749 139781364750208 model_lib_v2.py:652] Step 30100 per-step time 0.263s loss=1.367
    INFO:tensorflow:Step 30200 per-step time 0.252s loss=1.054
    I1009 07:57:22.899072 139781364750208 model_lib_v2.py:652] Step 30200 per-step time 0.252s loss=1.054
    INFO:tensorflow:Step 30300 per-step time 0.257s loss=0.935
    I1009 07:57:47.829975 139781364750208 model_lib_v2.py:652] Step 30300 per-step time 0.257s loss=0.935
    INFO:tensorflow:Step 30400 per-step time 0.250s loss=0.769
    I1009 07:58:12.864217 139781364750208 model_lib_v2.py:652] Step 30400 per-step time 0.250s loss=0.769
    INFO:tensorflow:Step 30500 per-step time 0.243s loss=0.865
    I1009 07:58:38.082502 139781364750208 model_lib_v2.py:652] Step 30500 per-step time 0.243s loss=0.865
    INFO:tensorflow:Step 30600 per-step time 0.247s loss=1.165
    I1009 07:59:03.287014 139781364750208 model_lib_v2.py:652] Step 30600 per-step time 0.247s loss=1.165
    INFO:tensorflow:Step 30700 per-step time 0.247s loss=0.968
    I1009 07:59:28.432365 139781364750208 model_lib_v2.py:652] Step 30700 per-step time 0.247s loss=0.968
    INFO:tensorflow:Step 30800 per-step time 0.267s loss=0.626
    I1009 07:59:53.767103 139781364750208 model_lib_v2.py:652] Step 30800 per-step time 0.267s loss=0.626
    INFO:tensorflow:Step 30900 per-step time 0.250s loss=0.861
    I1009 08:00:18.979584 139781364750208 model_lib_v2.py:652] Step 30900 per-step time 0.250s loss=0.861
    INFO:tensorflow:Step 31000 per-step time 0.245s loss=1.157
    I1009 08:00:44.245087 139781364750208 model_lib_v2.py:652] Step 31000 per-step time 0.245s loss=1.157
    INFO:tensorflow:Step 31100 per-step time 0.249s loss=0.794
    I1009 08:01:10.290127 139781364750208 model_lib_v2.py:652] Step 31100 per-step time 0.249s loss=0.794
    INFO:tensorflow:Step 31200 per-step time 0.267s loss=0.831
    I1009 08:01:35.392210 139781364750208 model_lib_v2.py:652] Step 31200 per-step time 0.267s loss=0.831
    INFO:tensorflow:Step 31300 per-step time 0.252s loss=0.634
    I1009 08:02:00.677732 139781364750208 model_lib_v2.py:652] Step 31300 per-step time 0.252s loss=0.634
    INFO:tensorflow:Step 31400 per-step time 0.260s loss=0.859
    I1009 08:02:25.771776 139781364750208 model_lib_v2.py:652] Step 31400 per-step time 0.260s loss=0.859
    INFO:tensorflow:Step 31500 per-step time 0.249s loss=0.854
    I1009 08:02:50.989642 139781364750208 model_lib_v2.py:652] Step 31500 per-step time 0.249s loss=0.854
    INFO:tensorflow:Step 31600 per-step time 0.250s loss=0.775
    I1009 08:03:16.107472 139781364750208 model_lib_v2.py:652] Step 31600 per-step time 0.250s loss=0.775
    INFO:tensorflow:Step 31700 per-step time 0.261s loss=0.880
    I1009 08:03:41.142423 139781364750208 model_lib_v2.py:652] Step 31700 per-step time 0.261s loss=0.880
    INFO:tensorflow:Step 31800 per-step time 0.242s loss=0.857
    I1009 08:04:06.069213 139781364750208 model_lib_v2.py:652] Step 31800 per-step time 0.242s loss=0.857
    INFO:tensorflow:Step 31900 per-step time 0.243s loss=0.921
    I1009 08:04:31.159446 139781364750208 model_lib_v2.py:652] Step 31900 per-step time 0.243s loss=0.921
    INFO:tensorflow:Step 32000 per-step time 0.256s loss=0.970
    I1009 08:04:56.300753 139781364750208 model_lib_v2.py:652] Step 32000 per-step time 0.256s loss=0.970
    INFO:tensorflow:Step 32100 per-step time 0.251s loss=1.205
    I1009 08:05:22.171967 139781364750208 model_lib_v2.py:652] Step 32100 per-step time 0.251s loss=1.205
    INFO:tensorflow:Step 32200 per-step time 0.253s loss=1.030
    I1009 08:05:47.514190 139781364750208 model_lib_v2.py:652] Step 32200 per-step time 0.253s loss=1.030
    INFO:tensorflow:Step 32300 per-step time 0.245s loss=0.901
    I1009 08:06:12.635100 139781364750208 model_lib_v2.py:652] Step 32300 per-step time 0.245s loss=0.901
    INFO:tensorflow:Step 32400 per-step time 0.248s loss=0.885
    I1009 08:06:37.648888 139781364750208 model_lib_v2.py:652] Step 32400 per-step time 0.248s loss=0.885
    INFO:tensorflow:Step 32500 per-step time 0.236s loss=0.944
    I1009 08:07:02.616171 139781364750208 model_lib_v2.py:652] Step 32500 per-step time 0.236s loss=0.944
    INFO:tensorflow:Step 32600 per-step time 0.246s loss=0.657
    I1009 08:07:27.486652 139781364750208 model_lib_v2.py:652] Step 32600 per-step time 0.246s loss=0.657
    INFO:tensorflow:Step 32700 per-step time 0.251s loss=0.792
    I1009 08:07:52.326669 139781364750208 model_lib_v2.py:652] Step 32700 per-step time 0.251s loss=0.792
    INFO:tensorflow:Step 32800 per-step time 0.267s loss=0.753
    I1009 08:08:17.222320 139781364750208 model_lib_v2.py:652] Step 32800 per-step time 0.267s loss=0.753
    INFO:tensorflow:Step 32900 per-step time 0.265s loss=0.907
    I1009 08:08:42.238437 139781364750208 model_lib_v2.py:652] Step 32900 per-step time 0.265s loss=0.907
    INFO:tensorflow:Step 33000 per-step time 0.260s loss=0.897
    I1009 08:09:07.137129 139781364750208 model_lib_v2.py:652] Step 33000 per-step time 0.260s loss=0.897
    INFO:tensorflow:Step 33100 per-step time 0.256s loss=0.886
    I1009 08:09:33.136149 139781364750208 model_lib_v2.py:652] Step 33100 per-step time 0.256s loss=0.886
    INFO:tensorflow:Step 33200 per-step time 0.258s loss=1.124
    I1009 08:09:58.221362 139781364750208 model_lib_v2.py:652] Step 33200 per-step time 0.258s loss=1.124
    INFO:tensorflow:Step 33300 per-step time 0.244s loss=0.783
    I1009 08:10:23.349076 139781364750208 model_lib_v2.py:652] Step 33300 per-step time 0.244s loss=0.783
    INFO:tensorflow:Step 33400 per-step time 0.261s loss=0.931
    I1009 08:10:48.132486 139781364750208 model_lib_v2.py:652] Step 33400 per-step time 0.261s loss=0.931
    INFO:tensorflow:Step 33500 per-step time 0.249s loss=0.597
    I1009 08:11:13.407766 139781364750208 model_lib_v2.py:652] Step 33500 per-step time 0.249s loss=0.597
    INFO:tensorflow:Step 33600 per-step time 0.246s loss=1.186
    I1009 08:11:38.529043 139781364750208 model_lib_v2.py:652] Step 33600 per-step time 0.246s loss=1.186
    INFO:tensorflow:Step 33700 per-step time 0.254s loss=0.855
    I1009 08:12:03.604651 139781364750208 model_lib_v2.py:652] Step 33700 per-step time 0.254s loss=0.855
    INFO:tensorflow:Step 33800 per-step time 0.255s loss=0.851
    I1009 08:12:28.545433 139781364750208 model_lib_v2.py:652] Step 33800 per-step time 0.255s loss=0.851
    INFO:tensorflow:Step 33900 per-step time 0.260s loss=0.737
    I1009 08:12:53.582155 139781364750208 model_lib_v2.py:652] Step 33900 per-step time 0.260s loss=0.737
    INFO:tensorflow:Step 34000 per-step time 0.247s loss=0.845
    I1009 08:13:18.585198 139781364750208 model_lib_v2.py:652] Step 34000 per-step time 0.247s loss=0.845
    INFO:tensorflow:Step 34100 per-step time 0.233s loss=0.933
    I1009 08:13:44.871116 139781364750208 model_lib_v2.py:652] Step 34100 per-step time 0.233s loss=0.933
    INFO:tensorflow:Step 34200 per-step time 0.242s loss=0.849
    I1009 08:14:09.747636 139781364750208 model_lib_v2.py:652] Step 34200 per-step time 0.242s loss=0.849
    INFO:tensorflow:Step 34300 per-step time 0.236s loss=0.863
    I1009 08:14:34.730039 139781364750208 model_lib_v2.py:652] Step 34300 per-step time 0.236s loss=0.863
    INFO:tensorflow:Step 34400 per-step time 0.251s loss=1.224
    I1009 08:14:59.662042 139781364750208 model_lib_v2.py:652] Step 34400 per-step time 0.251s loss=1.224
    INFO:tensorflow:Step 34500 per-step time 0.244s loss=1.115
    I1009 08:15:24.550219 139781364750208 model_lib_v2.py:652] Step 34500 per-step time 0.244s loss=1.115
    INFO:tensorflow:Step 34600 per-step time 0.257s loss=0.770
    I1009 08:15:49.533330 139781364750208 model_lib_v2.py:652] Step 34600 per-step time 0.257s loss=0.770
    INFO:tensorflow:Step 34700 per-step time 0.259s loss=1.030
    I1009 08:16:14.746023 139781364750208 model_lib_v2.py:652] Step 34700 per-step time 0.259s loss=1.030
    INFO:tensorflow:Step 34800 per-step time 0.258s loss=0.769
    I1009 08:16:39.546807 139781364750208 model_lib_v2.py:652] Step 34800 per-step time 0.258s loss=0.769
    INFO:tensorflow:Step 34900 per-step time 0.254s loss=0.762
    I1009 08:17:04.631398 139781364750208 model_lib_v2.py:652] Step 34900 per-step time 0.254s loss=0.762
    INFO:tensorflow:Step 35000 per-step time 0.251s loss=1.028
    I1009 08:17:29.470999 139781364750208 model_lib_v2.py:652] Step 35000 per-step time 0.251s loss=1.028
    INFO:tensorflow:Step 35100 per-step time 0.244s loss=0.925
    I1009 08:17:55.244432 139781364750208 model_lib_v2.py:652] Step 35100 per-step time 0.244s loss=0.925
    INFO:tensorflow:Step 35200 per-step time 0.237s loss=1.040
    I1009 08:18:20.114254 139781364750208 model_lib_v2.py:652] Step 35200 per-step time 0.237s loss=1.040
    INFO:tensorflow:Step 35300 per-step time 0.254s loss=1.019
    I1009 08:18:45.310862 139781364750208 model_lib_v2.py:652] Step 35300 per-step time 0.254s loss=1.019
    INFO:tensorflow:Step 35400 per-step time 0.237s loss=0.821
    I1009 08:19:10.198739 139781364750208 model_lib_v2.py:652] Step 35400 per-step time 0.237s loss=0.821
    INFO:tensorflow:Step 35500 per-step time 0.243s loss=0.947
    I1009 08:19:35.096735 139781364750208 model_lib_v2.py:652] Step 35500 per-step time 0.243s loss=0.947
    INFO:tensorflow:Step 35600 per-step time 0.249s loss=1.321
    I1009 08:19:59.894279 139781364750208 model_lib_v2.py:652] Step 35600 per-step time 0.249s loss=1.321
    INFO:tensorflow:Step 35700 per-step time 0.243s loss=0.555
    I1009 08:20:24.872899 139781364750208 model_lib_v2.py:652] Step 35700 per-step time 0.243s loss=0.555
    INFO:tensorflow:Step 35800 per-step time 0.237s loss=0.908
    I1009 08:20:49.733622 139781364750208 model_lib_v2.py:652] Step 35800 per-step time 0.237s loss=0.908
    INFO:tensorflow:Step 35900 per-step time 0.259s loss=1.161
    I1009 08:21:14.667440 139781364750208 model_lib_v2.py:652] Step 35900 per-step time 0.259s loss=1.161
    INFO:tensorflow:Step 36000 per-step time 0.255s loss=0.684
    I1009 08:21:39.882090 139781364750208 model_lib_v2.py:652] Step 36000 per-step time 0.255s loss=0.684
    INFO:tensorflow:Step 36100 per-step time 0.255s loss=0.957
    I1009 08:22:05.822545 139781364750208 model_lib_v2.py:652] Step 36100 per-step time 0.255s loss=0.957
    INFO:tensorflow:Step 36200 per-step time 0.237s loss=1.020
    I1009 08:22:30.571705 139781364750208 model_lib_v2.py:652] Step 36200 per-step time 0.237s loss=1.020
    INFO:tensorflow:Step 36300 per-step time 0.244s loss=1.050
    I1009 08:22:55.140676 139781364750208 model_lib_v2.py:652] Step 36300 per-step time 0.244s loss=1.050
    INFO:tensorflow:Step 36400 per-step time 0.250s loss=0.985
    I1009 08:23:19.851501 139781364750208 model_lib_v2.py:652] Step 36400 per-step time 0.250s loss=0.985
    INFO:tensorflow:Step 36500 per-step time 0.235s loss=1.277
    I1009 08:23:44.483041 139781364750208 model_lib_v2.py:652] Step 36500 per-step time 0.235s loss=1.277
    INFO:tensorflow:Step 36600 per-step time 0.240s loss=0.737
    I1009 08:24:09.054983 139781364750208 model_lib_v2.py:652] Step 36600 per-step time 0.240s loss=0.737
    INFO:tensorflow:Step 36700 per-step time 0.250s loss=1.086
    I1009 08:24:33.989174 139781364750208 model_lib_v2.py:652] Step 36700 per-step time 0.250s loss=1.086
    INFO:tensorflow:Step 36800 per-step time 0.246s loss=1.375
    I1009 08:24:58.951779 139781364750208 model_lib_v2.py:652] Step 36800 per-step time 0.246s loss=1.375
    INFO:tensorflow:Step 36900 per-step time 0.256s loss=1.068
    I1009 08:25:24.022097 139781364750208 model_lib_v2.py:652] Step 36900 per-step time 0.256s loss=1.068
    INFO:tensorflow:Step 37000 per-step time 0.246s loss=1.302
    I1009 08:25:48.940428 139781364750208 model_lib_v2.py:652] Step 37000 per-step time 0.246s loss=1.302
    INFO:tensorflow:Step 37100 per-step time 0.255s loss=0.863
    I1009 08:26:14.619603 139781364750208 model_lib_v2.py:652] Step 37100 per-step time 0.255s loss=0.863
    INFO:tensorflow:Step 37200 per-step time 0.248s loss=1.002
    I1009 08:26:39.792709 139781364750208 model_lib_v2.py:652] Step 37200 per-step time 0.248s loss=1.002
    INFO:tensorflow:Step 37300 per-step time 0.251s loss=1.193
    I1009 08:27:04.675811 139781364750208 model_lib_v2.py:652] Step 37300 per-step time 0.251s loss=1.193
    INFO:tensorflow:Step 37400 per-step time 0.245s loss=0.875
    I1009 08:27:29.975554 139781364750208 model_lib_v2.py:652] Step 37400 per-step time 0.245s loss=0.875
    INFO:tensorflow:Step 37500 per-step time 0.248s loss=1.071
    I1009 08:27:54.758098 139781364750208 model_lib_v2.py:652] Step 37500 per-step time 0.248s loss=1.071
    INFO:tensorflow:Step 37600 per-step time 0.246s loss=0.773
    I1009 08:28:19.750313 139781364750208 model_lib_v2.py:652] Step 37600 per-step time 0.246s loss=0.773
    INFO:tensorflow:Step 37700 per-step time 0.261s loss=1.084
    I1009 08:28:44.974598 139781364750208 model_lib_v2.py:652] Step 37700 per-step time 0.261s loss=1.084
    INFO:tensorflow:Step 37800 per-step time 0.253s loss=0.811
    I1009 08:29:09.676532 139781364750208 model_lib_v2.py:652] Step 37800 per-step time 0.253s loss=0.811
    INFO:tensorflow:Step 37900 per-step time 0.232s loss=1.051
    I1009 08:29:34.535442 139781364750208 model_lib_v2.py:652] Step 37900 per-step time 0.232s loss=1.051
    INFO:tensorflow:Step 38000 per-step time 0.236s loss=0.968
    I1009 08:29:59.183482 139781364750208 model_lib_v2.py:652] Step 38000 per-step time 0.236s loss=0.968
    INFO:tensorflow:Step 38100 per-step time 0.247s loss=0.866
    I1009 08:30:24.919515 139781364750208 model_lib_v2.py:652] Step 38100 per-step time 0.247s loss=0.866
    INFO:tensorflow:Step 38200 per-step time 0.246s loss=1.008
    I1009 08:30:49.796617 139781364750208 model_lib_v2.py:652] Step 38200 per-step time 0.246s loss=1.008
    INFO:tensorflow:Step 38300 per-step time 0.245s loss=0.807
    I1009 08:31:14.565752 139781364750208 model_lib_v2.py:652] Step 38300 per-step time 0.245s loss=0.807
    INFO:tensorflow:Step 38400 per-step time 0.255s loss=0.659
    I1009 08:31:39.509446 139781364750208 model_lib_v2.py:652] Step 38400 per-step time 0.255s loss=0.659
    INFO:tensorflow:Step 38500 per-step time 0.255s loss=0.783
    I1009 08:32:04.426761 139781364750208 model_lib_v2.py:652] Step 38500 per-step time 0.255s loss=0.783
    INFO:tensorflow:Step 38600 per-step time 0.245s loss=0.851
    I1009 08:32:29.318879 139781364750208 model_lib_v2.py:652] Step 38600 per-step time 0.245s loss=0.851
    INFO:tensorflow:Step 38700 per-step time 0.239s loss=0.948
    I1009 08:32:54.247678 139781364750208 model_lib_v2.py:652] Step 38700 per-step time 0.239s loss=0.948
    INFO:tensorflow:Step 38800 per-step time 0.256s loss=0.715
    I1009 08:33:19.107457 139781364750208 model_lib_v2.py:652] Step 38800 per-step time 0.256s loss=0.715
    INFO:tensorflow:Step 38900 per-step time 0.239s loss=0.852
    I1009 08:33:44.127336 139781364750208 model_lib_v2.py:652] Step 38900 per-step time 0.239s loss=0.852
    INFO:tensorflow:Step 39000 per-step time 0.233s loss=1.218
    I1009 08:34:08.997314 139781364750208 model_lib_v2.py:652] Step 39000 per-step time 0.233s loss=1.218
    INFO:tensorflow:Step 39100 per-step time 0.241s loss=1.096
    I1009 08:34:34.908798 139781364750208 model_lib_v2.py:652] Step 39100 per-step time 0.241s loss=1.096
    INFO:tensorflow:Step 39200 per-step time 0.252s loss=0.814
    I1009 08:34:59.847642 139781364750208 model_lib_v2.py:652] Step 39200 per-step time 0.252s loss=0.814
    INFO:tensorflow:Step 39300 per-step time 0.250s loss=1.016
    I1009 08:35:24.702100 139781364750208 model_lib_v2.py:652] Step 39300 per-step time 0.250s loss=1.016
    INFO:tensorflow:Step 39400 per-step time 0.255s loss=1.632
    I1009 08:35:49.539314 139781364750208 model_lib_v2.py:652] Step 39400 per-step time 0.255s loss=1.632
    INFO:tensorflow:Step 39500 per-step time 0.244s loss=0.871
    I1009 08:36:14.307147 139781364750208 model_lib_v2.py:652] Step 39500 per-step time 0.244s loss=0.871
    INFO:tensorflow:Step 39600 per-step time 0.250s loss=1.068
    I1009 08:36:39.307254 139781364750208 model_lib_v2.py:652] Step 39600 per-step time 0.250s loss=1.068
    INFO:tensorflow:Step 39700 per-step time 0.253s loss=0.884
    I1009 08:37:04.415092 139781364750208 model_lib_v2.py:652] Step 39700 per-step time 0.253s loss=0.884
    INFO:tensorflow:Step 39800 per-step time 0.263s loss=1.124
    I1009 08:37:29.337082 139781364750208 model_lib_v2.py:652] Step 39800 per-step time 0.263s loss=1.124
    INFO:tensorflow:Step 39900 per-step time 0.266s loss=0.645
    I1009 08:37:54.381438 139781364750208 model_lib_v2.py:652] Step 39900 per-step time 0.266s loss=0.645
    INFO:tensorflow:Step 40000 per-step time 0.249s loss=1.039
    I1009 08:38:19.351337 139781364750208 model_lib_v2.py:652] Step 40000 per-step time 0.249s loss=1.039
    INFO:tensorflow:Step 40100 per-step time 0.268s loss=0.980
    I1009 08:38:45.352738 139781364750208 model_lib_v2.py:652] Step 40100 per-step time 0.268s loss=0.980
    INFO:tensorflow:Step 40200 per-step time 0.261s loss=0.818
    I1009 08:39:10.417523 139781364750208 model_lib_v2.py:652] Step 40200 per-step time 0.261s loss=0.818
    INFO:tensorflow:Step 40300 per-step time 0.239s loss=0.601
    I1009 08:39:35.406464 139781364750208 model_lib_v2.py:652] Step 40300 per-step time 0.239s loss=0.601
    INFO:tensorflow:Step 40400 per-step time 0.246s loss=1.010
    I1009 08:40:00.559965 139781364750208 model_lib_v2.py:652] Step 40400 per-step time 0.246s loss=1.010
    INFO:tensorflow:Step 40500 per-step time 0.246s loss=0.968
    I1009 08:40:25.642105 139781364750208 model_lib_v2.py:652] Step 40500 per-step time 0.246s loss=0.968
    INFO:tensorflow:Step 40600 per-step time 0.254s loss=0.960
    I1009 08:40:50.753381 139781364750208 model_lib_v2.py:652] Step 40600 per-step time 0.254s loss=0.960
    INFO:tensorflow:Step 40700 per-step time 0.272s loss=0.778
    I1009 08:41:15.835160 139781364750208 model_lib_v2.py:652] Step 40700 per-step time 0.272s loss=0.778
    INFO:tensorflow:Step 40800 per-step time 0.244s loss=0.790
    I1009 08:41:40.931838 139781364750208 model_lib_v2.py:652] Step 40800 per-step time 0.244s loss=0.790
    INFO:tensorflow:Step 40900 per-step time 0.247s loss=0.852
    I1009 08:42:05.989413 139781364750208 model_lib_v2.py:652] Step 40900 per-step time 0.247s loss=0.852
    INFO:tensorflow:Step 41000 per-step time 0.248s loss=0.937
    I1009 08:42:30.944170 139781364750208 model_lib_v2.py:652] Step 41000 per-step time 0.248s loss=0.937
    INFO:tensorflow:Step 41100 per-step time 0.253s loss=0.845
    I1009 08:42:57.028474 139781364750208 model_lib_v2.py:652] Step 41100 per-step time 0.253s loss=0.845
    INFO:tensorflow:Step 41200 per-step time 0.250s loss=0.807
    I1009 08:43:21.983757 139781364750208 model_lib_v2.py:652] Step 41200 per-step time 0.250s loss=0.807
    INFO:tensorflow:Step 41300 per-step time 0.241s loss=1.148
    I1009 08:43:46.867622 139781364750208 model_lib_v2.py:652] Step 41300 per-step time 0.241s loss=1.148
    INFO:tensorflow:Step 41400 per-step time 0.260s loss=0.865
    I1009 08:44:11.894621 139781364750208 model_lib_v2.py:652] Step 41400 per-step time 0.260s loss=0.865
    INFO:tensorflow:Step 41500 per-step time 0.264s loss=1.086
    I1009 08:44:36.995985 139781364750208 model_lib_v2.py:652] Step 41500 per-step time 0.264s loss=1.086
    INFO:tensorflow:Step 41600 per-step time 0.258s loss=0.624
    I1009 08:45:02.163367 139781364750208 model_lib_v2.py:652] Step 41600 per-step time 0.258s loss=0.624
    INFO:tensorflow:Step 41700 per-step time 0.243s loss=0.813
    I1009 08:45:27.209073 139781364750208 model_lib_v2.py:652] Step 41700 per-step time 0.243s loss=0.813
    INFO:tensorflow:Step 41800 per-step time 0.252s loss=0.623
    I1009 08:45:52.418327 139781364750208 model_lib_v2.py:652] Step 41800 per-step time 0.252s loss=0.623
    INFO:tensorflow:Step 41900 per-step time 0.251s loss=0.728
    I1009 08:46:17.618077 139781364750208 model_lib_v2.py:652] Step 41900 per-step time 0.251s loss=0.728
    INFO:tensorflow:Step 42000 per-step time 0.257s loss=0.915
    I1009 08:46:43.014574 139781364750208 model_lib_v2.py:652] Step 42000 per-step time 0.257s loss=0.915
    INFO:tensorflow:Step 42100 per-step time 0.250s loss=1.135
    I1009 08:47:09.178107 139781364750208 model_lib_v2.py:652] Step 42100 per-step time 0.250s loss=1.135
    INFO:tensorflow:Step 42200 per-step time 0.267s loss=0.918
    I1009 08:47:34.618606 139781364750208 model_lib_v2.py:652] Step 42200 per-step time 0.267s loss=0.918
    INFO:tensorflow:Step 42300 per-step time 0.253s loss=0.667
    I1009 08:47:59.779205 139781364750208 model_lib_v2.py:652] Step 42300 per-step time 0.253s loss=0.667
    INFO:tensorflow:Step 42400 per-step time 0.262s loss=0.593
    I1009 08:48:24.727407 139781364750208 model_lib_v2.py:652] Step 42400 per-step time 0.262s loss=0.593
    INFO:tensorflow:Step 42500 per-step time 0.258s loss=0.784
    I1009 08:48:49.858295 139781364750208 model_lib_v2.py:652] Step 42500 per-step time 0.258s loss=0.784
    INFO:tensorflow:Step 42600 per-step time 0.249s loss=0.961
    I1009 08:49:15.117749 139781364750208 model_lib_v2.py:652] Step 42600 per-step time 0.249s loss=0.961
    INFO:tensorflow:Step 42700 per-step time 0.243s loss=0.781
    I1009 08:49:40.260683 139781364750208 model_lib_v2.py:652] Step 42700 per-step time 0.243s loss=0.781
    INFO:tensorflow:Step 42800 per-step time 0.252s loss=0.925
    I1009 08:50:05.255067 139781364750208 model_lib_v2.py:652] Step 42800 per-step time 0.252s loss=0.925
    INFO:tensorflow:Step 42900 per-step time 0.249s loss=1.083
    I1009 08:50:30.304924 139781364750208 model_lib_v2.py:652] Step 42900 per-step time 0.249s loss=1.083
    INFO:tensorflow:Step 43000 per-step time 0.255s loss=0.869
    I1009 08:50:55.394414 139781364750208 model_lib_v2.py:652] Step 43000 per-step time 0.255s loss=0.869
    INFO:tensorflow:Step 43100 per-step time 0.240s loss=0.882
    I1009 08:51:21.484104 139781364750208 model_lib_v2.py:652] Step 43100 per-step time 0.240s loss=0.882
    INFO:tensorflow:Step 43200 per-step time 0.264s loss=1.038
    I1009 08:51:46.605713 139781364750208 model_lib_v2.py:652] Step 43200 per-step time 0.264s loss=1.038
    INFO:tensorflow:Step 43300 per-step time 0.259s loss=0.860
    I1009 08:52:11.660026 139781364750208 model_lib_v2.py:652] Step 43300 per-step time 0.259s loss=0.860
    INFO:tensorflow:Step 43400 per-step time 0.251s loss=1.054
    I1009 08:52:37.115326 139781364750208 model_lib_v2.py:652] Step 43400 per-step time 0.251s loss=1.054
    INFO:tensorflow:Step 43500 per-step time 0.246s loss=0.756
    I1009 08:53:02.228493 139781364750208 model_lib_v2.py:652] Step 43500 per-step time 0.246s loss=0.756
    INFO:tensorflow:Step 43600 per-step time 0.242s loss=0.969
    I1009 08:53:27.289281 139781364750208 model_lib_v2.py:652] Step 43600 per-step time 0.242s loss=0.969
    INFO:tensorflow:Step 43700 per-step time 0.243s loss=0.655
    I1009 08:53:52.125488 139781364750208 model_lib_v2.py:652] Step 43700 per-step time 0.243s loss=0.655
    INFO:tensorflow:Step 43800 per-step time 0.244s loss=1.303
    I1009 08:54:16.829198 139781364750208 model_lib_v2.py:652] Step 43800 per-step time 0.244s loss=1.303
    INFO:tensorflow:Step 43900 per-step time 0.240s loss=1.133
    I1009 08:54:41.644403 139781364750208 model_lib_v2.py:652] Step 43900 per-step time 0.240s loss=1.133
    INFO:tensorflow:Step 44000 per-step time 0.252s loss=1.022
    I1009 08:55:06.416007 139781364750208 model_lib_v2.py:652] Step 44000 per-step time 0.252s loss=1.022
    INFO:tensorflow:Step 44100 per-step time 0.251s loss=0.696
    I1009 08:55:32.334639 139781364750208 model_lib_v2.py:652] Step 44100 per-step time 0.251s loss=0.696
    INFO:tensorflow:Step 44200 per-step time 0.247s loss=0.801
    I1009 08:55:57.352996 139781364750208 model_lib_v2.py:652] Step 44200 per-step time 0.247s loss=0.801
    INFO:tensorflow:Step 44300 per-step time 0.243s loss=0.614
    I1009 08:56:22.414180 139781364750208 model_lib_v2.py:652] Step 44300 per-step time 0.243s loss=0.614
    INFO:tensorflow:Step 44400 per-step time 0.238s loss=1.040
    I1009 08:56:47.358869 139781364750208 model_lib_v2.py:652] Step 44400 per-step time 0.238s loss=1.040
    INFO:tensorflow:Step 44500 per-step time 0.252s loss=0.914
    I1009 08:57:12.247711 139781364750208 model_lib_v2.py:652] Step 44500 per-step time 0.252s loss=0.914
    INFO:tensorflow:Step 44600 per-step time 0.270s loss=0.829
    I1009 08:57:37.370271 139781364750208 model_lib_v2.py:652] Step 44600 per-step time 0.270s loss=0.829
    INFO:tensorflow:Step 44700 per-step time 0.250s loss=0.776
    I1009 08:58:02.299103 139781364750208 model_lib_v2.py:652] Step 44700 per-step time 0.250s loss=0.776
    INFO:tensorflow:Step 44800 per-step time 0.254s loss=0.873
    I1009 08:58:27.067435 139781364750208 model_lib_v2.py:652] Step 44800 per-step time 0.254s loss=0.873
    INFO:tensorflow:Step 44900 per-step time 0.249s loss=1.453
    I1009 08:58:51.781713 139781364750208 model_lib_v2.py:652] Step 44900 per-step time 0.249s loss=1.453
    INFO:tensorflow:Step 45000 per-step time 0.246s loss=1.158
    I1009 08:59:16.428577 139781364750208 model_lib_v2.py:652] Step 45000 per-step time 0.246s loss=1.158
    INFO:tensorflow:Step 45100 per-step time 0.246s loss=1.004
    I1009 08:59:42.305094 139781364750208 model_lib_v2.py:652] Step 45100 per-step time 0.246s loss=1.004
    INFO:tensorflow:Step 45200 per-step time 0.246s loss=1.071
    I1009 09:00:07.002525 139781364750208 model_lib_v2.py:652] Step 45200 per-step time 0.246s loss=1.071
    INFO:tensorflow:Step 45300 per-step time 0.246s loss=0.746
    I1009 09:00:31.866070 139781364750208 model_lib_v2.py:652] Step 45300 per-step time 0.246s loss=0.746
    INFO:tensorflow:Step 45400 per-step time 0.255s loss=0.767
    I1009 09:00:56.641637 139781364750208 model_lib_v2.py:652] Step 45400 per-step time 0.255s loss=0.767
    INFO:tensorflow:Step 45500 per-step time 0.261s loss=0.994
    I1009 09:01:21.485188 139781364750208 model_lib_v2.py:652] Step 45500 per-step time 0.261s loss=0.994
    INFO:tensorflow:Step 45600 per-step time 0.261s loss=1.040
    I1009 09:01:46.437461 139781364750208 model_lib_v2.py:652] Step 45600 per-step time 0.261s loss=1.040
    INFO:tensorflow:Step 45700 per-step time 0.272s loss=1.022
    I1009 09:02:11.689435 139781364750208 model_lib_v2.py:652] Step 45700 per-step time 0.272s loss=1.022
    INFO:tensorflow:Step 45800 per-step time 0.266s loss=0.909
    I1009 09:02:36.814305 139781364750208 model_lib_v2.py:652] Step 45800 per-step time 0.266s loss=0.909
    INFO:tensorflow:Step 45900 per-step time 0.257s loss=1.107
    I1009 09:03:02.212052 139781364750208 model_lib_v2.py:652] Step 45900 per-step time 0.257s loss=1.107
    INFO:tensorflow:Step 46000 per-step time 0.250s loss=0.811
    I1009 09:03:27.451488 139781364750208 model_lib_v2.py:652] Step 46000 per-step time 0.250s loss=0.811
    INFO:tensorflow:Step 46100 per-step time 0.266s loss=0.785
    I1009 09:03:54.220506 139781364750208 model_lib_v2.py:652] Step 46100 per-step time 0.266s loss=0.785
    INFO:tensorflow:Step 46200 per-step time 0.257s loss=0.992
    I1009 09:04:19.302804 139781364750208 model_lib_v2.py:652] Step 46200 per-step time 0.257s loss=0.992
    INFO:tensorflow:Step 46300 per-step time 0.255s loss=0.803
    I1009 09:04:44.521833 139781364750208 model_lib_v2.py:652] Step 46300 per-step time 0.255s loss=0.803
    INFO:tensorflow:Step 46400 per-step time 0.268s loss=0.659
    I1009 09:05:10.023045 139781364750208 model_lib_v2.py:652] Step 46400 per-step time 0.268s loss=0.659
    INFO:tensorflow:Step 46500 per-step time 0.255s loss=0.737
    I1009 09:05:35.350045 139781364750208 model_lib_v2.py:652] Step 46500 per-step time 0.255s loss=0.737
    INFO:tensorflow:Step 46600 per-step time 0.240s loss=0.750
    I1009 09:06:00.533007 139781364750208 model_lib_v2.py:652] Step 46600 per-step time 0.240s loss=0.750
    INFO:tensorflow:Step 46700 per-step time 0.261s loss=0.665
    I1009 09:06:25.784800 139781364750208 model_lib_v2.py:652] Step 46700 per-step time 0.261s loss=0.665
    INFO:tensorflow:Step 46800 per-step time 0.238s loss=1.250
    I1009 09:06:51.010796 139781364750208 model_lib_v2.py:652] Step 46800 per-step time 0.238s loss=1.250
    INFO:tensorflow:Step 46900 per-step time 0.257s loss=0.831
    I1009 09:07:16.320364 139781364750208 model_lib_v2.py:652] Step 46900 per-step time 0.257s loss=0.831
    INFO:tensorflow:Step 47000 per-step time 0.248s loss=1.081
    I1009 09:07:41.567011 139781364750208 model_lib_v2.py:652] Step 47000 per-step time 0.248s loss=1.081
    INFO:tensorflow:Step 47100 per-step time 0.273s loss=0.837
    I1009 09:08:08.046808 139781364750208 model_lib_v2.py:652] Step 47100 per-step time 0.273s loss=0.837
    INFO:tensorflow:Step 47200 per-step time 0.245s loss=0.691
    I1009 09:08:33.360540 139781364750208 model_lib_v2.py:652] Step 47200 per-step time 0.245s loss=0.691
    INFO:tensorflow:Step 47300 per-step time 0.243s loss=0.895
    I1009 09:08:58.631509 139781364750208 model_lib_v2.py:652] Step 47300 per-step time 0.243s loss=0.895
    INFO:tensorflow:Step 47400 per-step time 0.246s loss=0.925
    I1009 09:09:23.861982 139781364750208 model_lib_v2.py:652] Step 47400 per-step time 0.246s loss=0.925
    INFO:tensorflow:Step 47500 per-step time 0.243s loss=0.996
    I1009 09:09:49.164225 139781364750208 model_lib_v2.py:652] Step 47500 per-step time 0.243s loss=0.996
    INFO:tensorflow:Step 47600 per-step time 0.243s loss=0.562
    I1009 09:10:14.384263 139781364750208 model_lib_v2.py:652] Step 47600 per-step time 0.243s loss=0.562
    INFO:tensorflow:Step 47700 per-step time 0.254s loss=0.813
    I1009 09:10:39.455494 139781364750208 model_lib_v2.py:652] Step 47700 per-step time 0.254s loss=0.813
    INFO:tensorflow:Step 47800 per-step time 0.268s loss=0.824
    I1009 09:11:04.703902 139781364750208 model_lib_v2.py:652] Step 47800 per-step time 0.268s loss=0.824
    INFO:tensorflow:Step 47900 per-step time 0.255s loss=0.916
    I1009 09:11:30.028984 139781364750208 model_lib_v2.py:652] Step 47900 per-step time 0.255s loss=0.916
    INFO:tensorflow:Step 48000 per-step time 0.252s loss=0.905
    I1009 09:11:55.347168 139781364750208 model_lib_v2.py:652] Step 48000 per-step time 0.252s loss=0.905
    INFO:tensorflow:Step 48100 per-step time 0.251s loss=0.913
    I1009 09:12:21.592712 139781364750208 model_lib_v2.py:652] Step 48100 per-step time 0.251s loss=0.913
    INFO:tensorflow:Step 48200 per-step time 0.264s loss=0.868
    I1009 09:12:46.992144 139781364750208 model_lib_v2.py:652] Step 48200 per-step time 0.264s loss=0.868
    INFO:tensorflow:Step 48300 per-step time 0.241s loss=0.572
    I1009 09:13:12.556582 139781364750208 model_lib_v2.py:652] Step 48300 per-step time 0.241s loss=0.572
    INFO:tensorflow:Step 48400 per-step time 0.258s loss=0.490
    I1009 09:13:37.865024 139781364750208 model_lib_v2.py:652] Step 48400 per-step time 0.258s loss=0.490
    INFO:tensorflow:Step 48500 per-step time 0.251s loss=0.905
    I1009 09:14:03.134108 139781364750208 model_lib_v2.py:652] Step 48500 per-step time 0.251s loss=0.905
    INFO:tensorflow:Step 48600 per-step time 0.251s loss=0.815
    I1009 09:14:28.427722 139781364750208 model_lib_v2.py:652] Step 48600 per-step time 0.251s loss=0.815
    INFO:tensorflow:Step 48700 per-step time 0.262s loss=0.820
    I1009 09:14:53.556611 139781364750208 model_lib_v2.py:652] Step 48700 per-step time 0.262s loss=0.820
    INFO:tensorflow:Step 48800 per-step time 0.241s loss=0.679
    I1009 09:15:18.863634 139781364750208 model_lib_v2.py:652] Step 48800 per-step time 0.241s loss=0.679
    INFO:tensorflow:Step 48900 per-step time 0.249s loss=0.706
    I1009 09:15:44.248753 139781364750208 model_lib_v2.py:652] Step 48900 per-step time 0.249s loss=0.706
    INFO:tensorflow:Step 49000 per-step time 0.253s loss=1.016
    I1009 09:16:09.481155 139781364750208 model_lib_v2.py:652] Step 49000 per-step time 0.253s loss=1.016
    INFO:tensorflow:Step 49100 per-step time 0.258s loss=1.214
    I1009 09:16:35.621562 139781364750208 model_lib_v2.py:652] Step 49100 per-step time 0.258s loss=1.214
    INFO:tensorflow:Step 49200 per-step time 0.246s loss=0.866
    I1009 09:17:00.926067 139781364750208 model_lib_v2.py:652] Step 49200 per-step time 0.246s loss=0.866
    INFO:tensorflow:Step 49300 per-step time 0.253s loss=1.081
    I1009 09:17:26.192183 139781364750208 model_lib_v2.py:652] Step 49300 per-step time 0.253s loss=1.081
    INFO:tensorflow:Step 49400 per-step time 0.258s loss=0.978
    I1009 09:17:51.550487 139781364750208 model_lib_v2.py:652] Step 49400 per-step time 0.258s loss=0.978
    INFO:tensorflow:Step 49500 per-step time 0.268s loss=0.799
    I1009 09:18:16.821256 139781364750208 model_lib_v2.py:652] Step 49500 per-step time 0.268s loss=0.799
    INFO:tensorflow:Step 49600 per-step time 0.243s loss=0.802
    I1009 09:18:42.142187 139781364750208 model_lib_v2.py:652] Step 49600 per-step time 0.243s loss=0.802
    INFO:tensorflow:Step 49700 per-step time 0.253s loss=1.021
    I1009 09:19:07.430824 139781364750208 model_lib_v2.py:652] Step 49700 per-step time 0.253s loss=1.021
    INFO:tensorflow:Step 49800 per-step time 0.252s loss=0.819
    I1009 09:19:32.783440 139781364750208 model_lib_v2.py:652] Step 49800 per-step time 0.252s loss=0.819
    INFO:tensorflow:Step 49900 per-step time 0.246s loss=1.001
    I1009 09:19:58.060085 139781364750208 model_lib_v2.py:652] Step 49900 per-step time 0.246s loss=1.001
    INFO:tensorflow:Step 50000 per-step time 0.245s loss=0.647
    I1009 09:20:23.198797 139781364750208 model_lib_v2.py:652] Step 50000 per-step time 0.245s loss=0.647
    INFO:tensorflow:Step 50100 per-step time 0.254s loss=1.077
    I1009 09:20:49.715039 139781364750208 model_lib_v2.py:652] Step 50100 per-step time 0.254s loss=1.077
    INFO:tensorflow:Step 50200 per-step time 0.260s loss=0.512
    I1009 09:21:15.112114 139781364750208 model_lib_v2.py:652] Step 50200 per-step time 0.260s loss=0.512
    INFO:tensorflow:Step 50300 per-step time 0.251s loss=0.882
    I1009 09:21:40.574521 139781364750208 model_lib_v2.py:652] Step 50300 per-step time 0.251s loss=0.882
    INFO:tensorflow:Step 50400 per-step time 0.251s loss=0.726
    I1009 09:22:05.901132 139781364750208 model_lib_v2.py:652] Step 50400 per-step time 0.251s loss=0.726
    INFO:tensorflow:Step 50500 per-step time 0.257s loss=1.039
    I1009 09:22:31.339565 139781364750208 model_lib_v2.py:652] Step 50500 per-step time 0.257s loss=1.039
    INFO:tensorflow:Step 50600 per-step time 0.249s loss=0.926
    I1009 09:22:56.526487 139781364750208 model_lib_v2.py:652] Step 50600 per-step time 0.249s loss=0.926
    INFO:tensorflow:Step 50700 per-step time 0.255s loss=0.952
    I1009 09:23:21.912928 139781364750208 model_lib_v2.py:652] Step 50700 per-step time 0.255s loss=0.952
    INFO:tensorflow:Step 50800 per-step time 0.238s loss=0.916
    I1009 09:23:47.454060 139781364750208 model_lib_v2.py:652] Step 50800 per-step time 0.238s loss=0.916
    INFO:tensorflow:Step 50900 per-step time 0.269s loss=0.905
    I1009 09:24:12.559841 139781364750208 model_lib_v2.py:652] Step 50900 per-step time 0.269s loss=0.905
    INFO:tensorflow:Step 51000 per-step time 0.273s loss=0.694
    I1009 09:24:37.903646 139781364750208 model_lib_v2.py:652] Step 51000 per-step time 0.273s loss=0.694
    INFO:tensorflow:Step 51100 per-step time 0.268s loss=1.160
    I1009 09:25:04.168357 139781364750208 model_lib_v2.py:652] Step 51100 per-step time 0.268s loss=1.160
    INFO:tensorflow:Step 51200 per-step time 0.258s loss=0.703
    I1009 09:25:29.530358 139781364750208 model_lib_v2.py:652] Step 51200 per-step time 0.258s loss=0.703
    INFO:tensorflow:Step 51300 per-step time 0.266s loss=0.728
    I1009 09:25:54.759258 139781364750208 model_lib_v2.py:652] Step 51300 per-step time 0.266s loss=0.728
    INFO:tensorflow:Step 51400 per-step time 0.254s loss=0.827
    I1009 09:26:20.090085 139781364750208 model_lib_v2.py:652] Step 51400 per-step time 0.254s loss=0.827
    INFO:tensorflow:Step 51500 per-step time 0.253s loss=0.695
    I1009 09:26:45.299437 139781364750208 model_lib_v2.py:652] Step 51500 per-step time 0.253s loss=0.695
    INFO:tensorflow:Step 51600 per-step time 0.239s loss=0.863
    I1009 09:27:10.680269 139781364750208 model_lib_v2.py:652] Step 51600 per-step time 0.239s loss=0.863
    INFO:tensorflow:Step 51700 per-step time 0.260s loss=0.708
    I1009 09:27:35.936190 139781364750208 model_lib_v2.py:652] Step 51700 per-step time 0.260s loss=0.708
    INFO:tensorflow:Step 51800 per-step time 0.261s loss=0.710
    I1009 09:28:01.213322 139781364750208 model_lib_v2.py:652] Step 51800 per-step time 0.261s loss=0.710
    INFO:tensorflow:Step 51900 per-step time 0.250s loss=0.662
    I1009 09:28:26.398434 139781364750208 model_lib_v2.py:652] Step 51900 per-step time 0.250s loss=0.662
    INFO:tensorflow:Step 52000 per-step time 0.249s loss=0.734
    I1009 09:28:51.915825 139781364750208 model_lib_v2.py:652] Step 52000 per-step time 0.249s loss=0.734
    INFO:tensorflow:Step 52100 per-step time 0.239s loss=0.865
    I1009 09:29:18.275710 139781364750208 model_lib_v2.py:652] Step 52100 per-step time 0.239s loss=0.865
    INFO:tensorflow:Step 52200 per-step time 0.248s loss=0.919
    I1009 09:29:43.469544 139781364750208 model_lib_v2.py:652] Step 52200 per-step time 0.248s loss=0.919
    INFO:tensorflow:Step 52300 per-step time 0.248s loss=0.701
    I1009 09:30:08.658382 139781364750208 model_lib_v2.py:652] Step 52300 per-step time 0.248s loss=0.701
    INFO:tensorflow:Step 52400 per-step time 0.250s loss=0.804
    I1009 09:30:33.882900 139781364750208 model_lib_v2.py:652] Step 52400 per-step time 0.250s loss=0.804
    INFO:tensorflow:Step 52500 per-step time 0.257s loss=0.424
    I1009 09:30:58.968441 139781364750208 model_lib_v2.py:652] Step 52500 per-step time 0.257s loss=0.424
    INFO:tensorflow:Step 52600 per-step time 0.249s loss=0.755
    I1009 09:31:24.151286 139781364750208 model_lib_v2.py:652] Step 52600 per-step time 0.249s loss=0.755
    INFO:tensorflow:Step 52700 per-step time 0.245s loss=0.782
    I1009 09:31:49.377051 139781364750208 model_lib_v2.py:652] Step 52700 per-step time 0.245s loss=0.782
    INFO:tensorflow:Step 52800 per-step time 0.253s loss=1.101
    I1009 09:32:14.378806 139781364750208 model_lib_v2.py:652] Step 52800 per-step time 0.253s loss=1.101
    INFO:tensorflow:Step 52900 per-step time 0.245s loss=0.745
    I1009 09:32:39.421985 139781364750208 model_lib_v2.py:652] Step 52900 per-step time 0.245s loss=0.745
    INFO:tensorflow:Step 53000 per-step time 0.262s loss=0.707
    I1009 09:33:04.599144 139781364750208 model_lib_v2.py:652] Step 53000 per-step time 0.262s loss=0.707
    INFO:tensorflow:Step 53100 per-step time 0.248s loss=0.850
    I1009 09:33:30.765318 139781364750208 model_lib_v2.py:652] Step 53100 per-step time 0.248s loss=0.850
    INFO:tensorflow:Step 53200 per-step time 0.248s loss=0.824
    I1009 09:33:56.000829 139781364750208 model_lib_v2.py:652] Step 53200 per-step time 0.248s loss=0.824
    INFO:tensorflow:Step 53300 per-step time 0.266s loss=0.711
    I1009 09:34:21.137730 139781364750208 model_lib_v2.py:652] Step 53300 per-step time 0.266s loss=0.711
    INFO:tensorflow:Step 53400 per-step time 0.244s loss=0.803
    I1009 09:34:46.137645 139781364750208 model_lib_v2.py:652] Step 53400 per-step time 0.244s loss=0.803
    INFO:tensorflow:Step 53500 per-step time 0.251s loss=0.903
    I1009 09:35:11.307373 139781364750208 model_lib_v2.py:652] Step 53500 per-step time 0.251s loss=0.903
    INFO:tensorflow:Step 53600 per-step time 0.246s loss=0.634
    I1009 09:35:36.310786 139781364750208 model_lib_v2.py:652] Step 53600 per-step time 0.246s loss=0.634
    INFO:tensorflow:Step 53700 per-step time 0.242s loss=1.161
    I1009 09:36:01.343211 139781364750208 model_lib_v2.py:652] Step 53700 per-step time 0.242s loss=1.161
    INFO:tensorflow:Step 53800 per-step time 0.248s loss=0.780
    I1009 09:36:26.336928 139781364750208 model_lib_v2.py:652] Step 53800 per-step time 0.248s loss=0.780
    INFO:tensorflow:Step 53900 per-step time 0.252s loss=0.689
    I1009 09:36:51.392245 139781364750208 model_lib_v2.py:652] Step 53900 per-step time 0.252s loss=0.689
    INFO:tensorflow:Step 54000 per-step time 0.239s loss=0.692
    I1009 09:37:16.422869 139781364750208 model_lib_v2.py:652] Step 54000 per-step time 0.239s loss=0.692
    INFO:tensorflow:Step 54100 per-step time 0.253s loss=0.735
    I1009 09:37:42.505376 139781364750208 model_lib_v2.py:652] Step 54100 per-step time 0.253s loss=0.735
    INFO:tensorflow:Step 54200 per-step time 0.243s loss=1.122
    I1009 09:38:07.605219 139781364750208 model_lib_v2.py:652] Step 54200 per-step time 0.243s loss=1.122
    INFO:tensorflow:Step 54300 per-step time 0.250s loss=0.637
    I1009 09:38:32.602161 139781364750208 model_lib_v2.py:652] Step 54300 per-step time 0.250s loss=0.637
    INFO:tensorflow:Step 54400 per-step time 0.265s loss=0.941
    I1009 09:38:57.652631 139781364750208 model_lib_v2.py:652] Step 54400 per-step time 0.265s loss=0.941
    INFO:tensorflow:Step 54500 per-step time 0.247s loss=0.723
    I1009 09:39:22.919429 139781364750208 model_lib_v2.py:652] Step 54500 per-step time 0.247s loss=0.723
    INFO:tensorflow:Step 54600 per-step time 0.257s loss=0.877
    I1009 09:39:48.069302 139781364750208 model_lib_v2.py:652] Step 54600 per-step time 0.257s loss=0.877
    INFO:tensorflow:Step 54700 per-step time 0.255s loss=0.901
    I1009 09:40:13.035591 139781364750208 model_lib_v2.py:652] Step 54700 per-step time 0.255s loss=0.901
    INFO:tensorflow:Step 54800 per-step time 0.254s loss=0.954
    I1009 09:40:38.048756 139781364750208 model_lib_v2.py:652] Step 54800 per-step time 0.254s loss=0.954
    INFO:tensorflow:Step 54900 per-step time 0.254s loss=0.632
    I1009 09:41:03.226469 139781364750208 model_lib_v2.py:652] Step 54900 per-step time 0.254s loss=0.632
    INFO:tensorflow:Step 55000 per-step time 0.265s loss=1.048
    I1009 09:41:28.259480 139781364750208 model_lib_v2.py:652] Step 55000 per-step time 0.265s loss=1.048
    INFO:tensorflow:Step 55100 per-step time 0.247s loss=0.886
    I1009 09:41:54.254026 139781364750208 model_lib_v2.py:652] Step 55100 per-step time 0.247s loss=0.886
    INFO:tensorflow:Step 55200 per-step time 0.246s loss=0.694
    I1009 09:42:19.394620 139781364750208 model_lib_v2.py:652] Step 55200 per-step time 0.246s loss=0.694
    INFO:tensorflow:Step 55300 per-step time 0.246s loss=0.832
    I1009 09:42:44.424937 139781364750208 model_lib_v2.py:652] Step 55300 per-step time 0.246s loss=0.832
    INFO:tensorflow:Step 55400 per-step time 0.245s loss=0.826
    I1009 09:43:09.500432 139781364750208 model_lib_v2.py:652] Step 55400 per-step time 0.245s loss=0.826
    INFO:tensorflow:Step 55500 per-step time 0.252s loss=0.789
    I1009 09:43:34.749015 139781364750208 model_lib_v2.py:652] Step 55500 per-step time 0.252s loss=0.789
    INFO:tensorflow:Step 55600 per-step time 0.259s loss=0.798
    I1009 09:43:59.726231 139781364750208 model_lib_v2.py:652] Step 55600 per-step time 0.259s loss=0.798
    INFO:tensorflow:Step 55700 per-step time 0.239s loss=1.030
    I1009 09:44:24.926445 139781364750208 model_lib_v2.py:652] Step 55700 per-step time 0.239s loss=1.030
    INFO:tensorflow:Step 55800 per-step time 0.252s loss=1.098
    I1009 09:44:49.858263 139781364750208 model_lib_v2.py:652] Step 55800 per-step time 0.252s loss=1.098
    INFO:tensorflow:Step 55900 per-step time 0.272s loss=1.139
    I1009 09:45:14.815138 139781364750208 model_lib_v2.py:652] Step 55900 per-step time 0.272s loss=1.139
    INFO:tensorflow:Step 56000 per-step time 0.248s loss=0.928
    I1009 09:45:39.845204 139781364750208 model_lib_v2.py:652] Step 56000 per-step time 0.248s loss=0.928
    INFO:tensorflow:Step 56100 per-step time 0.247s loss=1.028
    I1009 09:46:05.924604 139781364750208 model_lib_v2.py:652] Step 56100 per-step time 0.247s loss=1.028
    INFO:tensorflow:Step 56200 per-step time 0.244s loss=0.918
    I1009 09:46:30.970465 139781364750208 model_lib_v2.py:652] Step 56200 per-step time 0.244s loss=0.918
    INFO:tensorflow:Step 56300 per-step time 0.248s loss=0.916
    I1009 09:46:55.910588 139781364750208 model_lib_v2.py:652] Step 56300 per-step time 0.248s loss=0.916
    INFO:tensorflow:Step 56400 per-step time 0.255s loss=0.751
    I1009 09:47:20.819909 139781364750208 model_lib_v2.py:652] Step 56400 per-step time 0.255s loss=0.751
    INFO:tensorflow:Step 56500 per-step time 0.243s loss=1.010
    I1009 09:47:45.933188 139781364750208 model_lib_v2.py:652] Step 56500 per-step time 0.243s loss=1.010
    INFO:tensorflow:Step 56600 per-step time 0.236s loss=0.761
    I1009 09:48:10.794004 139781364750208 model_lib_v2.py:652] Step 56600 per-step time 0.236s loss=0.761
    INFO:tensorflow:Step 56700 per-step time 0.248s loss=0.667
    I1009 09:48:35.743877 139781364750208 model_lib_v2.py:652] Step 56700 per-step time 0.248s loss=0.667
    INFO:tensorflow:Step 56800 per-step time 0.255s loss=0.625
    I1009 09:49:00.764645 139781364750208 model_lib_v2.py:652] Step 56800 per-step time 0.255s loss=0.625
    INFO:tensorflow:Step 56900 per-step time 0.272s loss=0.855
    I1009 09:49:25.820311 139781364750208 model_lib_v2.py:652] Step 56900 per-step time 0.272s loss=0.855
    INFO:tensorflow:Step 57000 per-step time 0.243s loss=0.651
    I1009 09:49:50.694643 139781364750208 model_lib_v2.py:652] Step 57000 per-step time 0.243s loss=0.651
    INFO:tensorflow:Step 57100 per-step time 0.243s loss=0.666
    I1009 09:50:17.419063 139781364750208 model_lib_v2.py:652] Step 57100 per-step time 0.243s loss=0.666
    INFO:tensorflow:Step 57200 per-step time 0.245s loss=1.137
    I1009 09:50:42.544858 139781364750208 model_lib_v2.py:652] Step 57200 per-step time 0.245s loss=1.137
    INFO:tensorflow:Step 57300 per-step time 0.254s loss=0.773
    I1009 09:51:07.519177 139781364750208 model_lib_v2.py:652] Step 57300 per-step time 0.254s loss=0.773
    INFO:tensorflow:Step 57400 per-step time 0.240s loss=0.766
    I1009 09:51:32.460167 139781364750208 model_lib_v2.py:652] Step 57400 per-step time 0.240s loss=0.766
    INFO:tensorflow:Step 57500 per-step time 0.245s loss=0.942
    I1009 09:51:57.611799 139781364750208 model_lib_v2.py:652] Step 57500 per-step time 0.245s loss=0.942
    INFO:tensorflow:Step 57600 per-step time 0.264s loss=0.691
    I1009 09:52:22.748911 139781364750208 model_lib_v2.py:652] Step 57600 per-step time 0.264s loss=0.691
    INFO:tensorflow:Step 57700 per-step time 0.245s loss=0.652
    I1009 09:52:47.709387 139781364750208 model_lib_v2.py:652] Step 57700 per-step time 0.245s loss=0.652
    INFO:tensorflow:Step 57800 per-step time 0.235s loss=1.109
    I1009 09:53:12.648945 139781364750208 model_lib_v2.py:652] Step 57800 per-step time 0.235s loss=1.109
    INFO:tensorflow:Step 57900 per-step time 0.244s loss=0.719
    I1009 09:53:37.542280 139781364750208 model_lib_v2.py:652] Step 57900 per-step time 0.244s loss=0.719
    INFO:tensorflow:Step 58000 per-step time 0.256s loss=1.007
    I1009 09:54:02.385605 139781364750208 model_lib_v2.py:652] Step 58000 per-step time 0.256s loss=1.007
    INFO:tensorflow:Step 58100 per-step time 0.242s loss=1.121
    I1009 09:54:28.609889 139781364750208 model_lib_v2.py:652] Step 58100 per-step time 0.242s loss=1.121
    INFO:tensorflow:Step 58200 per-step time 0.254s loss=1.155
    I1009 09:54:53.837800 139781364750208 model_lib_v2.py:652] Step 58200 per-step time 0.254s loss=1.155
    INFO:tensorflow:Step 58300 per-step time 0.252s loss=1.201
    I1009 09:55:18.872164 139781364750208 model_lib_v2.py:652] Step 58300 per-step time 0.252s loss=1.201
    INFO:tensorflow:Step 58400 per-step time 0.244s loss=0.587
    I1009 09:55:43.801543 139781364750208 model_lib_v2.py:652] Step 58400 per-step time 0.244s loss=0.587
    INFO:tensorflow:Step 58500 per-step time 0.246s loss=0.904
    I1009 09:56:08.684129 139781364750208 model_lib_v2.py:652] Step 58500 per-step time 0.246s loss=0.904
    INFO:tensorflow:Step 58600 per-step time 0.254s loss=0.822
    I1009 09:56:33.574151 139781364750208 model_lib_v2.py:652] Step 58600 per-step time 0.254s loss=0.822
    INFO:tensorflow:Step 58700 per-step time 0.235s loss=0.983
    I1009 09:56:58.568603 139781364750208 model_lib_v2.py:652] Step 58700 per-step time 0.235s loss=0.983
    INFO:tensorflow:Step 58800 per-step time 0.250s loss=0.927
    I1009 09:57:23.652662 139781364750208 model_lib_v2.py:652] Step 58800 per-step time 0.250s loss=0.927
    INFO:tensorflow:Step 58900 per-step time 0.251s loss=0.999
    I1009 09:57:48.459848 139781364750208 model_lib_v2.py:652] Step 58900 per-step time 0.251s loss=0.999
    INFO:tensorflow:Step 59000 per-step time 0.239s loss=0.697
    I1009 09:58:13.498646 139781364750208 model_lib_v2.py:652] Step 59000 per-step time 0.239s loss=0.697
    INFO:tensorflow:Step 59100 per-step time 0.254s loss=1.090
    I1009 09:58:39.418752 139781364750208 model_lib_v2.py:652] Step 59100 per-step time 0.254s loss=1.090
    INFO:tensorflow:Step 59200 per-step time 0.257s loss=1.155
    I1009 09:59:04.321661 139781364750208 model_lib_v2.py:652] Step 59200 per-step time 0.257s loss=1.155
    INFO:tensorflow:Step 59300 per-step time 0.246s loss=0.679
    I1009 09:59:29.274481 139781364750208 model_lib_v2.py:652] Step 59300 per-step time 0.246s loss=0.679
    INFO:tensorflow:Step 59400 per-step time 0.247s loss=0.975
    I1009 09:59:54.313341 139781364750208 model_lib_v2.py:652] Step 59400 per-step time 0.247s loss=0.975
    INFO:tensorflow:Step 59500 per-step time 0.256s loss=0.957
    I1009 10:00:18.993836 139781364750208 model_lib_v2.py:652] Step 59500 per-step time 0.256s loss=0.957
    INFO:tensorflow:Step 59600 per-step time 0.244s loss=0.523
    I1009 10:00:43.629395 139781364750208 model_lib_v2.py:652] Step 59600 per-step time 0.244s loss=0.523
    INFO:tensorflow:Step 59700 per-step time 0.250s loss=0.898
    I1009 10:01:08.325670 139781364750208 model_lib_v2.py:652] Step 59700 per-step time 0.250s loss=0.898
    INFO:tensorflow:Step 59800 per-step time 0.243s loss=0.674
    I1009 10:01:33.032215 139781364750208 model_lib_v2.py:652] Step 59800 per-step time 0.243s loss=0.674
    INFO:tensorflow:Step 59900 per-step time 0.256s loss=0.826
    I1009 10:01:57.766819 139781364750208 model_lib_v2.py:652] Step 59900 per-step time 0.256s loss=0.826
    INFO:tensorflow:Step 60000 per-step time 0.245s loss=0.891
    I1009 10:02:22.571713 139781364750208 model_lib_v2.py:652] Step 60000 per-step time 0.245s loss=0.891
    INFO:tensorflow:Step 60100 per-step time 0.242s loss=1.089
    I1009 10:02:48.726887 139781364750208 model_lib_v2.py:652] Step 60100 per-step time 0.242s loss=1.089
    INFO:tensorflow:Step 60200 per-step time 0.249s loss=0.555
    I1009 10:03:13.408513 139781364750208 model_lib_v2.py:652] Step 60200 per-step time 0.249s loss=0.555
    INFO:tensorflow:Step 60300 per-step time 0.261s loss=0.979
    I1009 10:03:38.137622 139781364750208 model_lib_v2.py:652] Step 60300 per-step time 0.261s loss=0.979
    INFO:tensorflow:Step 60400 per-step time 0.253s loss=0.970
    I1009 10:04:03.095659 139781364750208 model_lib_v2.py:652] Step 60400 per-step time 0.253s loss=0.970
    INFO:tensorflow:Step 60500 per-step time 0.249s loss=0.553
    I1009 10:04:27.871215 139781364750208 model_lib_v2.py:652] Step 60500 per-step time 0.249s loss=0.553
    INFO:tensorflow:Step 60600 per-step time 0.248s loss=0.752
    I1009 10:04:52.656022 139781364750208 model_lib_v2.py:652] Step 60600 per-step time 0.248s loss=0.752
    INFO:tensorflow:Step 60700 per-step time 0.239s loss=0.843
    I1009 10:05:17.586149 139781364750208 model_lib_v2.py:652] Step 60700 per-step time 0.239s loss=0.843
    INFO:tensorflow:Step 60800 per-step time 0.243s loss=0.886
    I1009 10:05:42.284168 139781364750208 model_lib_v2.py:652] Step 60800 per-step time 0.243s loss=0.886
    INFO:tensorflow:Step 60900 per-step time 0.256s loss=0.839
    I1009 10:06:07.035906 139781364750208 model_lib_v2.py:652] Step 60900 per-step time 0.256s loss=0.839
    INFO:tensorflow:Step 61000 per-step time 0.244s loss=0.819
    I1009 10:06:31.673098 139781364750208 model_lib_v2.py:652] Step 61000 per-step time 0.244s loss=0.819
    INFO:tensorflow:Step 61100 per-step time 0.252s loss=0.711
    I1009 10:06:57.247537 139781364750208 model_lib_v2.py:652] Step 61100 per-step time 0.252s loss=0.711
    INFO:tensorflow:Step 61200 per-step time 0.244s loss=1.254
    I1009 10:07:21.952286 139781364750208 model_lib_v2.py:652] Step 61200 per-step time 0.244s loss=1.254
    INFO:tensorflow:Step 61300 per-step time 0.251s loss=0.691
    I1009 10:07:46.600334 139781364750208 model_lib_v2.py:652] Step 61300 per-step time 0.251s loss=0.691
    INFO:tensorflow:Step 61400 per-step time 0.251s loss=1.115
    I1009 10:08:11.397084 139781364750208 model_lib_v2.py:652] Step 61400 per-step time 0.251s loss=1.115
    INFO:tensorflow:Step 61500 per-step time 0.257s loss=0.822
    I1009 10:08:36.160857 139781364750208 model_lib_v2.py:652] Step 61500 per-step time 0.257s loss=0.822
    INFO:tensorflow:Step 61600 per-step time 0.250s loss=1.105
    I1009 10:09:00.988991 139781364750208 model_lib_v2.py:652] Step 61600 per-step time 0.250s loss=1.105
    INFO:tensorflow:Step 61700 per-step time 0.235s loss=0.906
    I1009 10:09:25.681470 139781364750208 model_lib_v2.py:652] Step 61700 per-step time 0.235s loss=0.906
    INFO:tensorflow:Step 61800 per-step time 0.244s loss=0.811
    I1009 10:09:50.339823 139781364750208 model_lib_v2.py:652] Step 61800 per-step time 0.244s loss=0.811
    INFO:tensorflow:Step 61900 per-step time 0.264s loss=1.163
    I1009 10:10:15.255975 139781364750208 model_lib_v2.py:652] Step 61900 per-step time 0.264s loss=1.163
    INFO:tensorflow:Step 62000 per-step time 0.248s loss=1.066
    I1009 10:10:39.937129 139781364750208 model_lib_v2.py:652] Step 62000 per-step time 0.248s loss=1.066
    INFO:tensorflow:Step 62100 per-step time 0.250s loss=0.598
    I1009 10:11:05.841671 139781364750208 model_lib_v2.py:652] Step 62100 per-step time 0.250s loss=0.598
    INFO:tensorflow:Step 62200 per-step time 0.250s loss=1.348
    I1009 10:11:30.533023 139781364750208 model_lib_v2.py:652] Step 62200 per-step time 0.250s loss=1.348
    INFO:tensorflow:Step 62300 per-step time 0.246s loss=1.361
    I1009 10:11:55.411004 139781364750208 model_lib_v2.py:652] Step 62300 per-step time 0.246s loss=1.361
    INFO:tensorflow:Step 62400 per-step time 0.251s loss=0.818
    I1009 10:12:20.383908 139781364750208 model_lib_v2.py:652] Step 62400 per-step time 0.251s loss=0.818
    INFO:tensorflow:Step 62500 per-step time 0.248s loss=0.778
    I1009 10:12:45.139503 139781364750208 model_lib_v2.py:652] Step 62500 per-step time 0.248s loss=0.778
    INFO:tensorflow:Step 62600 per-step time 0.248s loss=0.504
    I1009 10:13:09.741728 139781364750208 model_lib_v2.py:652] Step 62600 per-step time 0.248s loss=0.504
    INFO:tensorflow:Step 62700 per-step time 0.232s loss=0.646
    I1009 10:13:34.498058 139781364750208 model_lib_v2.py:652] Step 62700 per-step time 0.232s loss=0.646
    INFO:tensorflow:Step 62800 per-step time 0.251s loss=0.901
    I1009 10:13:59.260277 139781364750208 model_lib_v2.py:652] Step 62800 per-step time 0.251s loss=0.901
    INFO:tensorflow:Step 62900 per-step time 0.250s loss=0.915
    I1009 10:14:24.087545 139781364750208 model_lib_v2.py:652] Step 62900 per-step time 0.250s loss=0.915
    INFO:tensorflow:Step 63000 per-step time 0.235s loss=0.690
    I1009 10:14:48.897443 139781364750208 model_lib_v2.py:652] Step 63000 per-step time 0.235s loss=0.690
    INFO:tensorflow:Step 63100 per-step time 0.250s loss=1.175
    I1009 10:15:14.729120 139781364750208 model_lib_v2.py:652] Step 63100 per-step time 0.250s loss=1.175
    INFO:tensorflow:Step 63200 per-step time 0.254s loss=0.952
    I1009 10:15:39.995175 139781364750208 model_lib_v2.py:652] Step 63200 per-step time 0.254s loss=0.952
    INFO:tensorflow:Step 63300 per-step time 0.239s loss=0.664
    I1009 10:16:05.010876 139781364750208 model_lib_v2.py:652] Step 63300 per-step time 0.239s loss=0.664
    INFO:tensorflow:Step 63400 per-step time 0.248s loss=0.581
    I1009 10:16:29.863236 139781364750208 model_lib_v2.py:652] Step 63400 per-step time 0.248s loss=0.581
    INFO:tensorflow:Step 63500 per-step time 0.247s loss=0.689
    I1009 10:16:54.824056 139781364750208 model_lib_v2.py:652] Step 63500 per-step time 0.247s loss=0.689
    INFO:tensorflow:Step 63600 per-step time 0.253s loss=0.929
    I1009 10:17:19.819165 139781364750208 model_lib_v2.py:652] Step 63600 per-step time 0.253s loss=0.929
    INFO:tensorflow:Step 63700 per-step time 0.260s loss=0.779
    I1009 10:17:44.630342 139781364750208 model_lib_v2.py:652] Step 63700 per-step time 0.260s loss=0.779
    INFO:tensorflow:Step 63800 per-step time 0.245s loss=0.800
    I1009 10:18:09.462701 139781364750208 model_lib_v2.py:652] Step 63800 per-step time 0.245s loss=0.800
    INFO:tensorflow:Step 63900 per-step time 0.252s loss=0.730
    I1009 10:18:34.371202 139781364750208 model_lib_v2.py:652] Step 63900 per-step time 0.252s loss=0.730
    INFO:tensorflow:Step 64000 per-step time 0.253s loss=0.787
    I1009 10:18:59.302441 139781364750208 model_lib_v2.py:652] Step 64000 per-step time 0.253s loss=0.787
    INFO:tensorflow:Step 64100 per-step time 0.250s loss=0.789
    I1009 10:19:25.053247 139781364750208 model_lib_v2.py:652] Step 64100 per-step time 0.250s loss=0.789
    INFO:tensorflow:Step 64200 per-step time 0.244s loss=0.894
    I1009 10:19:49.856786 139781364750208 model_lib_v2.py:652] Step 64200 per-step time 0.244s loss=0.894
    INFO:tensorflow:Step 64300 per-step time 0.261s loss=0.678
    I1009 10:20:14.684453 139781364750208 model_lib_v2.py:652] Step 64300 per-step time 0.261s loss=0.678
    INFO:tensorflow:Step 64400 per-step time 0.248s loss=0.758
    I1009 10:20:39.797944 139781364750208 model_lib_v2.py:652] Step 64400 per-step time 0.248s loss=0.758
    INFO:tensorflow:Step 64500 per-step time 0.249s loss=1.119
    I1009 10:21:04.893739 139781364750208 model_lib_v2.py:652] Step 64500 per-step time 0.249s loss=1.119
    INFO:tensorflow:Step 64600 per-step time 0.252s loss=0.857
    I1009 10:21:29.733054 139781364750208 model_lib_v2.py:652] Step 64600 per-step time 0.252s loss=0.857
    INFO:tensorflow:Step 64700 per-step time 0.252s loss=0.567
    I1009 10:21:54.483641 139781364750208 model_lib_v2.py:652] Step 64700 per-step time 0.252s loss=0.567
    INFO:tensorflow:Step 64800 per-step time 0.237s loss=0.760
    I1009 10:22:19.299170 139781364750208 model_lib_v2.py:652] Step 64800 per-step time 0.237s loss=0.760
    INFO:tensorflow:Step 64900 per-step time 0.245s loss=0.964
    I1009 10:22:44.088669 139781364750208 model_lib_v2.py:652] Step 64900 per-step time 0.245s loss=0.964
    INFO:tensorflow:Step 65000 per-step time 0.240s loss=0.759
    I1009 10:23:08.808621 139781364750208 model_lib_v2.py:652] Step 65000 per-step time 0.240s loss=0.759
    INFO:tensorflow:Step 65100 per-step time 0.247s loss=0.802
    I1009 10:23:34.420009 139781364750208 model_lib_v2.py:652] Step 65100 per-step time 0.247s loss=0.802
    INFO:tensorflow:Step 65200 per-step time 0.245s loss=0.939
    I1009 10:23:59.243825 139781364750208 model_lib_v2.py:652] Step 65200 per-step time 0.245s loss=0.939
    INFO:tensorflow:Step 65300 per-step time 0.249s loss=0.758
    I1009 10:24:23.936075 139781364750208 model_lib_v2.py:652] Step 65300 per-step time 0.249s loss=0.758
    INFO:tensorflow:Step 65400 per-step time 0.239s loss=1.074
    I1009 10:24:48.771337 139781364750208 model_lib_v2.py:652] Step 65400 per-step time 0.239s loss=1.074
    INFO:tensorflow:Step 65500 per-step time 0.248s loss=0.889
    I1009 10:25:13.618116 139781364750208 model_lib_v2.py:652] Step 65500 per-step time 0.248s loss=0.889
    INFO:tensorflow:Step 65600 per-step time 0.260s loss=0.933
    I1009 10:25:38.487232 139781364750208 model_lib_v2.py:652] Step 65600 per-step time 0.260s loss=0.933
    INFO:tensorflow:Step 65700 per-step time 0.240s loss=0.716
    I1009 10:26:03.465332 139781364750208 model_lib_v2.py:652] Step 65700 per-step time 0.240s loss=0.716
    INFO:tensorflow:Step 65800 per-step time 0.251s loss=0.806
    I1009 10:26:28.345928 139781364750208 model_lib_v2.py:652] Step 65800 per-step time 0.251s loss=0.806
    INFO:tensorflow:Step 65900 per-step time 0.266s loss=1.058
    I1009 10:26:53.117208 139781364750208 model_lib_v2.py:652] Step 65900 per-step time 0.266s loss=1.058
    INFO:tensorflow:Step 66000 per-step time 0.244s loss=0.868
    I1009 10:27:17.938524 139781364750208 model_lib_v2.py:652] Step 66000 per-step time 0.244s loss=0.868
    INFO:tensorflow:Step 66100 per-step time 0.263s loss=0.867
    I1009 10:27:43.618315 139781364750208 model_lib_v2.py:652] Step 66100 per-step time 0.263s loss=0.867
    INFO:tensorflow:Step 66200 per-step time 0.241s loss=1.054
    I1009 10:28:08.251613 139781364750208 model_lib_v2.py:652] Step 66200 per-step time 0.241s loss=1.054
    INFO:tensorflow:Step 66300 per-step time 0.258s loss=0.520
    I1009 10:28:33.040220 139781364750208 model_lib_v2.py:652] Step 66300 per-step time 0.258s loss=0.520
    INFO:tensorflow:Step 66400 per-step time 0.239s loss=0.707
    I1009 10:28:57.588212 139781364750208 model_lib_v2.py:652] Step 66400 per-step time 0.239s loss=0.707
    INFO:tensorflow:Step 66500 per-step time 0.239s loss=0.730
    I1009 10:29:22.340064 139781364750208 model_lib_v2.py:652] Step 66500 per-step time 0.239s loss=0.730
    INFO:tensorflow:Step 66600 per-step time 0.246s loss=0.721
    I1009 10:29:47.019150 139781364750208 model_lib_v2.py:652] Step 66600 per-step time 0.246s loss=0.721
    INFO:tensorflow:Step 66700 per-step time 0.239s loss=1.065
    I1009 10:30:11.669443 139781364750208 model_lib_v2.py:652] Step 66700 per-step time 0.239s loss=1.065
    INFO:tensorflow:Step 66800 per-step time 0.241s loss=0.988
    I1009 10:30:36.514060 139781364750208 model_lib_v2.py:652] Step 66800 per-step time 0.241s loss=0.988
    INFO:tensorflow:Step 66900 per-step time 0.259s loss=0.625
    I1009 10:31:01.579337 139781364750208 model_lib_v2.py:652] Step 66900 per-step time 0.259s loss=0.625
    INFO:tensorflow:Step 67000 per-step time 0.249s loss=0.789
    I1009 10:31:26.297145 139781364750208 model_lib_v2.py:652] Step 67000 per-step time 0.249s loss=0.789
    INFO:tensorflow:Step 67100 per-step time 0.251s loss=0.682
    I1009 10:31:52.025671 139781364750208 model_lib_v2.py:652] Step 67100 per-step time 0.251s loss=0.682
    INFO:tensorflow:Step 67200 per-step time 0.252s loss=0.857
    I1009 10:32:16.775496 139781364750208 model_lib_v2.py:652] Step 67200 per-step time 0.252s loss=0.857
    INFO:tensorflow:Step 67300 per-step time 0.251s loss=0.418
    I1009 10:32:41.606895 139781364750208 model_lib_v2.py:652] Step 67300 per-step time 0.251s loss=0.418
    INFO:tensorflow:Step 67400 per-step time 0.235s loss=0.800
    I1009 10:33:06.380338 139781364750208 model_lib_v2.py:652] Step 67400 per-step time 0.235s loss=0.800
    INFO:tensorflow:Step 67500 per-step time 0.244s loss=0.819
    I1009 10:33:31.134704 139781364750208 model_lib_v2.py:652] Step 67500 per-step time 0.244s loss=0.819
    INFO:tensorflow:Step 67600 per-step time 0.238s loss=0.624
    I1009 10:33:55.967867 139781364750208 model_lib_v2.py:652] Step 67600 per-step time 0.238s loss=0.624
    INFO:tensorflow:Step 67700 per-step time 0.250s loss=0.848
    I1009 10:34:20.720807 139781364750208 model_lib_v2.py:652] Step 67700 per-step time 0.250s loss=0.848
    INFO:tensorflow:Step 67800 per-step time 0.253s loss=1.145
    I1009 10:34:45.822872 139781364750208 model_lib_v2.py:652] Step 67800 per-step time 0.253s loss=1.145
    INFO:tensorflow:Step 67900 per-step time 0.257s loss=0.964
    I1009 10:35:10.886934 139781364750208 model_lib_v2.py:652] Step 67900 per-step time 0.257s loss=0.964
    INFO:tensorflow:Step 68000 per-step time 0.242s loss=0.688
    I1009 10:35:35.771000 139781364750208 model_lib_v2.py:652] Step 68000 per-step time 0.242s loss=0.688
    INFO:tensorflow:Step 68100 per-step time 0.255s loss=0.814
    I1009 10:36:01.669934 139781364750208 model_lib_v2.py:652] Step 68100 per-step time 0.255s loss=0.814
    INFO:tensorflow:Step 68200 per-step time 0.241s loss=0.801
    I1009 10:36:26.611382 139781364750208 model_lib_v2.py:652] Step 68200 per-step time 0.241s loss=0.801
    INFO:tensorflow:Step 68300 per-step time 0.244s loss=0.661
    I1009 10:36:51.390538 139781364750208 model_lib_v2.py:652] Step 68300 per-step time 0.244s loss=0.661
    INFO:tensorflow:Step 68400 per-step time 0.243s loss=0.821
    I1009 10:37:16.199392 139781364750208 model_lib_v2.py:652] Step 68400 per-step time 0.243s loss=0.821
    INFO:tensorflow:Step 68500 per-step time 0.257s loss=0.651
    I1009 10:37:41.062096 139781364750208 model_lib_v2.py:652] Step 68500 per-step time 0.257s loss=0.651
    INFO:tensorflow:Step 68600 per-step time 0.247s loss=1.207
    I1009 10:38:05.769196 139781364750208 model_lib_v2.py:652] Step 68600 per-step time 0.247s loss=1.207
    INFO:tensorflow:Step 68700 per-step time 0.252s loss=0.854
    I1009 10:38:30.518841 139781364750208 model_lib_v2.py:652] Step 68700 per-step time 0.252s loss=0.854
    INFO:tensorflow:Step 68800 per-step time 0.254s loss=1.233
    I1009 10:38:55.243248 139781364750208 model_lib_v2.py:652] Step 68800 per-step time 0.254s loss=1.233
    INFO:tensorflow:Step 68900 per-step time 0.248s loss=0.815
    I1009 10:39:19.939335 139781364750208 model_lib_v2.py:652] Step 68900 per-step time 0.248s loss=0.815
    INFO:tensorflow:Step 69000 per-step time 0.241s loss=0.817
    I1009 10:39:44.761945 139781364750208 model_lib_v2.py:652] Step 69000 per-step time 0.241s loss=0.817
    INFO:tensorflow:Step 69100 per-step time 0.232s loss=0.473
    I1009 10:40:10.414595 139781364750208 model_lib_v2.py:652] Step 69100 per-step time 0.232s loss=0.473
    INFO:tensorflow:Step 69200 per-step time 0.241s loss=0.982
    I1009 10:40:35.062999 139781364750208 model_lib_v2.py:652] Step 69200 per-step time 0.241s loss=0.982
    INFO:tensorflow:Step 69300 per-step time 0.242s loss=1.110
    I1009 10:40:59.893778 139781364750208 model_lib_v2.py:652] Step 69300 per-step time 0.242s loss=1.110
    INFO:tensorflow:Step 69400 per-step time 0.245s loss=0.533
    I1009 10:41:24.894114 139781364750208 model_lib_v2.py:652] Step 69400 per-step time 0.245s loss=0.533
    INFO:tensorflow:Step 69500 per-step time 0.245s loss=0.431
    I1009 10:41:49.790864 139781364750208 model_lib_v2.py:652] Step 69500 per-step time 0.245s loss=0.431
    INFO:tensorflow:Step 69600 per-step time 0.246s loss=0.758
    I1009 10:42:14.593412 139781364750208 model_lib_v2.py:652] Step 69600 per-step time 0.246s loss=0.758
    INFO:tensorflow:Step 69700 per-step time 0.242s loss=0.797
    I1009 10:42:39.328680 139781364750208 model_lib_v2.py:652] Step 69700 per-step time 0.242s loss=0.797
    INFO:tensorflow:Step 69800 per-step time 0.256s loss=1.019
    I1009 10:43:04.089317 139781364750208 model_lib_v2.py:652] Step 69800 per-step time 0.256s loss=1.019
    INFO:tensorflow:Step 69900 per-step time 0.253s loss=1.126
    I1009 10:43:28.804156 139781364750208 model_lib_v2.py:652] Step 69900 per-step time 0.253s loss=1.126
    INFO:tensorflow:Step 70000 per-step time 0.243s loss=0.752
    I1009 10:43:53.332144 139781364750208 model_lib_v2.py:652] Step 70000 per-step time 0.243s loss=0.752
    INFO:tensorflow:Step 70100 per-step time 0.251s loss=1.048
    I1009 10:44:19.038389 139781364750208 model_lib_v2.py:652] Step 70100 per-step time 0.251s loss=1.048
    INFO:tensorflow:Step 70200 per-step time 0.242s loss=0.897
    I1009 10:44:43.908482 139781364750208 model_lib_v2.py:652] Step 70200 per-step time 0.242s loss=0.897
    INFO:tensorflow:Step 70300 per-step time 0.254s loss=0.894
    I1009 10:45:08.668190 139781364750208 model_lib_v2.py:652] Step 70300 per-step time 0.254s loss=0.894
    INFO:tensorflow:Step 70400 per-step time 0.233s loss=0.667
    I1009 10:45:33.296134 139781364750208 model_lib_v2.py:652] Step 70400 per-step time 0.233s loss=0.667
    INFO:tensorflow:Step 70500 per-step time 0.236s loss=0.634
    I1009 10:45:57.957165 139781364750208 model_lib_v2.py:652] Step 70500 per-step time 0.236s loss=0.634
    INFO:tensorflow:Step 70600 per-step time 0.252s loss=0.640
    I1009 10:46:22.700101 139781364750208 model_lib_v2.py:652] Step 70600 per-step time 0.252s loss=0.640
    INFO:tensorflow:Step 70700 per-step time 0.255s loss=1.329
    I1009 10:46:47.853053 139781364750208 model_lib_v2.py:652] Step 70700 per-step time 0.255s loss=1.329
    INFO:tensorflow:Step 70800 per-step time 0.254s loss=0.692
    I1009 10:47:12.688648 139781364750208 model_lib_v2.py:652] Step 70800 per-step time 0.254s loss=0.692
    INFO:tensorflow:Step 70900 per-step time 0.253s loss=0.551
    I1009 10:47:37.435716 139781364750208 model_lib_v2.py:652] Step 70900 per-step time 0.253s loss=0.551
    INFO:tensorflow:Step 71000 per-step time 0.242s loss=0.861
    I1009 10:48:02.475849 139781364750208 model_lib_v2.py:652] Step 71000 per-step time 0.242s loss=0.861
    INFO:tensorflow:Step 71100 per-step time 0.243s loss=1.159
    I1009 10:48:28.190904 139781364750208 model_lib_v2.py:652] Step 71100 per-step time 0.243s loss=1.159
    INFO:tensorflow:Step 71200 per-step time 0.246s loss=0.593
    I1009 10:48:53.077516 139781364750208 model_lib_v2.py:652] Step 71200 per-step time 0.246s loss=0.593
    INFO:tensorflow:Step 71300 per-step time 0.236s loss=0.519
    I1009 10:49:17.715500 139781364750208 model_lib_v2.py:652] Step 71300 per-step time 0.236s loss=0.519
    INFO:tensorflow:Step 71400 per-step time 0.242s loss=1.006
    I1009 10:49:42.363265 139781364750208 model_lib_v2.py:652] Step 71400 per-step time 0.242s loss=1.006
    INFO:tensorflow:Step 71500 per-step time 0.250s loss=0.812
    I1009 10:50:07.084315 139781364750208 model_lib_v2.py:652] Step 71500 per-step time 0.250s loss=0.812
    INFO:tensorflow:Step 71600 per-step time 0.259s loss=0.579
    I1009 10:50:31.886518 139781364750208 model_lib_v2.py:652] Step 71600 per-step time 0.259s loss=0.579
    INFO:tensorflow:Step 71700 per-step time 0.252s loss=0.787
    I1009 10:50:56.657217 139781364750208 model_lib_v2.py:652] Step 71700 per-step time 0.252s loss=0.787
    INFO:tensorflow:Step 71800 per-step time 0.238s loss=1.082
    I1009 10:51:21.349001 139781364750208 model_lib_v2.py:652] Step 71800 per-step time 0.238s loss=1.082
    INFO:tensorflow:Step 71900 per-step time 0.247s loss=0.671
    I1009 10:51:46.360608 139781364750208 model_lib_v2.py:652] Step 71900 per-step time 0.247s loss=0.671
    INFO:tensorflow:Step 72000 per-step time 0.244s loss=0.883
    I1009 10:52:11.181447 139781364750208 model_lib_v2.py:652] Step 72000 per-step time 0.244s loss=0.883
    INFO:tensorflow:Step 72100 per-step time 0.257s loss=0.657
    I1009 10:52:36.859583 139781364750208 model_lib_v2.py:652] Step 72100 per-step time 0.257s loss=0.657
    INFO:tensorflow:Step 72200 per-step time 0.249s loss=0.959
    I1009 10:53:01.804645 139781364750208 model_lib_v2.py:652] Step 72200 per-step time 0.249s loss=0.959
    INFO:tensorflow:Step 72300 per-step time 0.239s loss=0.790
    I1009 10:53:26.583726 139781364750208 model_lib_v2.py:652] Step 72300 per-step time 0.239s loss=0.790
    INFO:tensorflow:Step 72400 per-step time 0.251s loss=0.505
    I1009 10:53:51.416165 139781364750208 model_lib_v2.py:652] Step 72400 per-step time 0.251s loss=0.505
    INFO:tensorflow:Step 72500 per-step time 0.241s loss=0.607
    I1009 10:54:16.142854 139781364750208 model_lib_v2.py:652] Step 72500 per-step time 0.241s loss=0.607
    INFO:tensorflow:Step 72600 per-step time 0.243s loss=0.861
    I1009 10:54:40.917339 139781364750208 model_lib_v2.py:652] Step 72600 per-step time 0.243s loss=0.861
    INFO:tensorflow:Step 72700 per-step time 0.242s loss=0.757
    I1009 10:55:05.804317 139781364750208 model_lib_v2.py:652] Step 72700 per-step time 0.242s loss=0.757
    INFO:tensorflow:Step 72800 per-step time 0.250s loss=0.510
    I1009 10:55:30.516160 139781364750208 model_lib_v2.py:652] Step 72800 per-step time 0.250s loss=0.510
    INFO:tensorflow:Step 72900 per-step time 0.236s loss=1.405
    I1009 10:55:55.180278 139781364750208 model_lib_v2.py:652] Step 72900 per-step time 0.236s loss=1.405
    INFO:tensorflow:Step 73000 per-step time 0.246s loss=0.726
    I1009 10:56:19.732601 139781364750208 model_lib_v2.py:652] Step 73000 per-step time 0.246s loss=0.726
    INFO:tensorflow:Step 73100 per-step time 0.255s loss=0.995
    I1009 10:56:45.182257 139781364750208 model_lib_v2.py:652] Step 73100 per-step time 0.255s loss=0.995
    INFO:tensorflow:Step 73200 per-step time 0.246s loss=0.634
    I1009 10:57:09.971011 139781364750208 model_lib_v2.py:652] Step 73200 per-step time 0.246s loss=0.634
    INFO:tensorflow:Step 73300 per-step time 0.254s loss=0.889
    I1009 10:57:34.359875 139781364750208 model_lib_v2.py:652] Step 73300 per-step time 0.254s loss=0.889
    INFO:tensorflow:Step 73400 per-step time 0.239s loss=0.835
    I1009 10:57:58.744438 139781364750208 model_lib_v2.py:652] Step 73400 per-step time 0.239s loss=0.835
    INFO:tensorflow:Step 73500 per-step time 0.240s loss=1.263
    I1009 10:58:23.178116 139781364750208 model_lib_v2.py:652] Step 73500 per-step time 0.240s loss=1.263
    INFO:tensorflow:Step 73600 per-step time 0.238s loss=0.923
    I1009 10:58:47.576163 139781364750208 model_lib_v2.py:652] Step 73600 per-step time 0.238s loss=0.923
    INFO:tensorflow:Step 73700 per-step time 0.247s loss=0.496
    I1009 10:59:12.149943 139781364750208 model_lib_v2.py:652] Step 73700 per-step time 0.247s loss=0.496
    INFO:tensorflow:Step 73800 per-step time 0.249s loss=1.099
    I1009 10:59:36.916115 139781364750208 model_lib_v2.py:652] Step 73800 per-step time 0.249s loss=1.099
    INFO:tensorflow:Step 73900 per-step time 0.237s loss=0.658
    I1009 11:00:01.618630 139781364750208 model_lib_v2.py:652] Step 73900 per-step time 0.237s loss=0.658
    INFO:tensorflow:Step 74000 per-step time 0.247s loss=0.703
    I1009 11:00:26.209980 139781364750208 model_lib_v2.py:652] Step 74000 per-step time 0.247s loss=0.703
    INFO:tensorflow:Step 74100 per-step time 0.249s loss=1.025
    I1009 11:00:51.608755 139781364750208 model_lib_v2.py:652] Step 74100 per-step time 0.249s loss=1.025
    INFO:tensorflow:Step 74200 per-step time 0.253s loss=1.040
    I1009 11:01:16.193907 139781364750208 model_lib_v2.py:652] Step 74200 per-step time 0.253s loss=1.040
    INFO:tensorflow:Step 74300 per-step time 0.245s loss=0.824
    I1009 11:01:40.881108 139781364750208 model_lib_v2.py:652] Step 74300 per-step time 0.245s loss=0.824
    INFO:tensorflow:Step 74400 per-step time 0.248s loss=0.986
    I1009 11:02:06.069552 139781364750208 model_lib_v2.py:652] Step 74400 per-step time 0.248s loss=0.986
    INFO:tensorflow:Step 74500 per-step time 0.249s loss=0.564
    I1009 11:02:30.755751 139781364750208 model_lib_v2.py:652] Step 74500 per-step time 0.249s loss=0.564
    INFO:tensorflow:Step 74600 per-step time 0.248s loss=0.544
    I1009 11:02:55.366838 139781364750208 model_lib_v2.py:652] Step 74600 per-step time 0.248s loss=0.544
    INFO:tensorflow:Step 74700 per-step time 0.250s loss=0.652
    I1009 11:03:19.826100 139781364750208 model_lib_v2.py:652] Step 74700 per-step time 0.250s loss=0.652
    INFO:tensorflow:Step 74800 per-step time 0.235s loss=0.727
    I1009 11:03:44.402010 139781364750208 model_lib_v2.py:652] Step 74800 per-step time 0.235s loss=0.727
    INFO:tensorflow:Step 74900 per-step time 0.239s loss=1.274
    I1009 11:04:08.710518 139781364750208 model_lib_v2.py:652] Step 74900 per-step time 0.239s loss=1.274
    INFO:tensorflow:Step 75000 per-step time 0.243s loss=0.766
    I1009 11:04:32.946620 139781364750208 model_lib_v2.py:652] Step 75000 per-step time 0.243s loss=0.766
    INFO:tensorflow:Step 75100 per-step time 0.251s loss=0.767
    I1009 11:04:58.667206 139781364750208 model_lib_v2.py:652] Step 75100 per-step time 0.251s loss=0.767
    INFO:tensorflow:Step 75200 per-step time 0.247s loss=0.982
    I1009 11:05:23.326045 139781364750208 model_lib_v2.py:652] Step 75200 per-step time 0.247s loss=0.982
    INFO:tensorflow:Step 75300 per-step time 0.258s loss=0.513
    I1009 11:05:47.973132 139781364750208 model_lib_v2.py:652] Step 75300 per-step time 0.258s loss=0.513
    INFO:tensorflow:Step 75400 per-step time 0.246s loss=0.810
    I1009 11:06:12.591282 139781364750208 model_lib_v2.py:652] Step 75400 per-step time 0.246s loss=0.810
    INFO:tensorflow:Step 75500 per-step time 0.254s loss=1.004
    I1009 11:06:37.194522 139781364750208 model_lib_v2.py:652] Step 75500 per-step time 0.254s loss=1.004
    INFO:tensorflow:Step 75600 per-step time 0.249s loss=0.716
    I1009 11:07:01.982944 139781364750208 model_lib_v2.py:652] Step 75600 per-step time 0.249s loss=0.716
    INFO:tensorflow:Step 75700 per-step time 0.254s loss=0.942
    I1009 11:07:26.935712 139781364750208 model_lib_v2.py:652] Step 75700 per-step time 0.254s loss=0.942
    INFO:tensorflow:Step 75800 per-step time 0.243s loss=0.520
    I1009 11:07:51.732671 139781364750208 model_lib_v2.py:652] Step 75800 per-step time 0.243s loss=0.520
    INFO:tensorflow:Step 75900 per-step time 0.247s loss=0.626
    I1009 11:08:16.679445 139781364750208 model_lib_v2.py:652] Step 75900 per-step time 0.247s loss=0.626
    INFO:tensorflow:Step 76000 per-step time 0.251s loss=0.627
    I1009 11:08:41.341136 139781364750208 model_lib_v2.py:652] Step 76000 per-step time 0.251s loss=0.627
    INFO:tensorflow:Step 76100 per-step time 0.238s loss=0.805
    I1009 11:09:07.029805 139781364750208 model_lib_v2.py:652] Step 76100 per-step time 0.238s loss=0.805
    INFO:tensorflow:Step 76200 per-step time 0.251s loss=0.840
    I1009 11:09:31.722031 139781364750208 model_lib_v2.py:652] Step 76200 per-step time 0.251s loss=0.840
    INFO:tensorflow:Step 76300 per-step time 0.247s loss=0.823
    I1009 11:09:56.631412 139781364750208 model_lib_v2.py:652] Step 76300 per-step time 0.247s loss=0.823
    INFO:tensorflow:Step 76400 per-step time 0.241s loss=0.692
    I1009 11:10:21.320501 139781364750208 model_lib_v2.py:652] Step 76400 per-step time 0.241s loss=0.692
    INFO:tensorflow:Step 76500 per-step time 0.249s loss=0.633
    I1009 11:10:46.218152 139781364750208 model_lib_v2.py:652] Step 76500 per-step time 0.249s loss=0.633
    INFO:tensorflow:Step 76600 per-step time 0.245s loss=0.735
    I1009 11:11:10.889975 139781364750208 model_lib_v2.py:652] Step 76600 per-step time 0.245s loss=0.735
    INFO:tensorflow:Step 76700 per-step time 0.238s loss=0.675
    I1009 11:11:35.616021 139781364750208 model_lib_v2.py:652] Step 76700 per-step time 0.238s loss=0.675
    INFO:tensorflow:Step 76800 per-step time 0.242s loss=1.249
    I1009 11:12:00.287655 139781364750208 model_lib_v2.py:652] Step 76800 per-step time 0.242s loss=1.249
    INFO:tensorflow:Step 76900 per-step time 0.242s loss=0.772
    I1009 11:12:25.028799 139781364750208 model_lib_v2.py:652] Step 76900 per-step time 0.242s loss=0.772
    INFO:tensorflow:Step 77000 per-step time 0.249s loss=0.976
    I1009 11:12:49.709181 139781364750208 model_lib_v2.py:652] Step 77000 per-step time 0.249s loss=0.976
    INFO:tensorflow:Step 77100 per-step time 0.244s loss=0.654
    I1009 11:13:15.396704 139781364750208 model_lib_v2.py:652] Step 77100 per-step time 0.244s loss=0.654
    INFO:tensorflow:Step 77200 per-step time 0.253s loss=0.735
    I1009 11:13:39.972255 139781364750208 model_lib_v2.py:652] Step 77200 per-step time 0.253s loss=0.735
    INFO:tensorflow:Step 77300 per-step time 0.240s loss=0.434
    I1009 11:14:04.620015 139781364750208 model_lib_v2.py:652] Step 77300 per-step time 0.240s loss=0.434
    INFO:tensorflow:Step 77400 per-step time 0.240s loss=0.652
    I1009 11:14:29.310194 139781364750208 model_lib_v2.py:652] Step 77400 per-step time 0.240s loss=0.652
    INFO:tensorflow:Step 77500 per-step time 0.250s loss=0.934
    I1009 11:14:54.110955 139781364750208 model_lib_v2.py:652] Step 77500 per-step time 0.250s loss=0.934
    INFO:tensorflow:Step 77600 per-step time 0.240s loss=0.892
    I1009 11:15:18.993494 139781364750208 model_lib_v2.py:652] Step 77600 per-step time 0.240s loss=0.892
    INFO:tensorflow:Step 77700 per-step time 0.244s loss=0.629
    I1009 11:15:43.689121 139781364750208 model_lib_v2.py:652] Step 77700 per-step time 0.244s loss=0.629
    INFO:tensorflow:Step 77800 per-step time 0.255s loss=0.917
    I1009 11:16:08.464825 139781364750208 model_lib_v2.py:652] Step 77800 per-step time 0.255s loss=0.917
    INFO:tensorflow:Step 77900 per-step time 0.241s loss=0.815
    I1009 11:16:33.099016 139781364750208 model_lib_v2.py:652] Step 77900 per-step time 0.241s loss=0.815
    INFO:tensorflow:Step 78000 per-step time 0.252s loss=0.873
    I1009 11:16:57.663223 139781364750208 model_lib_v2.py:652] Step 78000 per-step time 0.252s loss=0.873
    INFO:tensorflow:Step 78100 per-step time 0.233s loss=0.492
    I1009 11:17:23.315283 139781364750208 model_lib_v2.py:652] Step 78100 per-step time 0.233s loss=0.492
    INFO:tensorflow:Step 78200 per-step time 0.250s loss=0.569
    I1009 11:17:48.428318 139781364750208 model_lib_v2.py:652] Step 78200 per-step time 0.250s loss=0.569
    INFO:tensorflow:Step 78300 per-step time 0.251s loss=0.546
    I1009 11:18:13.141289 139781364750208 model_lib_v2.py:652] Step 78300 per-step time 0.251s loss=0.546
    INFO:tensorflow:Step 78400 per-step time 0.246s loss=0.926
    I1009 11:18:37.895184 139781364750208 model_lib_v2.py:652] Step 78400 per-step time 0.246s loss=0.926
    INFO:tensorflow:Step 78500 per-step time 0.257s loss=0.973
    I1009 11:19:02.623743 139781364750208 model_lib_v2.py:652] Step 78500 per-step time 0.257s loss=0.973
    INFO:tensorflow:Step 78600 per-step time 0.244s loss=1.054
    I1009 11:19:27.357085 139781364750208 model_lib_v2.py:652] Step 78600 per-step time 0.244s loss=1.054
    INFO:tensorflow:Step 78700 per-step time 0.249s loss=0.690
    I1009 11:19:52.209228 139781364750208 model_lib_v2.py:652] Step 78700 per-step time 0.249s loss=0.690
    INFO:tensorflow:Step 78800 per-step time 0.246s loss=0.651
    I1009 11:20:17.030590 139781364750208 model_lib_v2.py:652] Step 78800 per-step time 0.246s loss=0.651
    INFO:tensorflow:Step 78900 per-step time 0.241s loss=0.935
    I1009 11:20:41.805403 139781364750208 model_lib_v2.py:652] Step 78900 per-step time 0.241s loss=0.935
    INFO:tensorflow:Step 79000 per-step time 0.249s loss=0.789
    I1009 11:21:06.502807 139781364750208 model_lib_v2.py:652] Step 79000 per-step time 0.249s loss=0.789
    INFO:tensorflow:Step 79100 per-step time 0.256s loss=0.994
    I1009 11:21:32.237486 139781364750208 model_lib_v2.py:652] Step 79100 per-step time 0.256s loss=0.994
    INFO:tensorflow:Step 79200 per-step time 0.258s loss=1.180
    I1009 11:21:56.874726 139781364750208 model_lib_v2.py:652] Step 79200 per-step time 0.258s loss=1.180
    INFO:tensorflow:Step 79300 per-step time 0.251s loss=0.825
    I1009 11:22:21.583593 139781364750208 model_lib_v2.py:652] Step 79300 per-step time 0.251s loss=0.825
    INFO:tensorflow:Step 79400 per-step time 0.256s loss=0.869
    I1009 11:22:46.522703 139781364750208 model_lib_v2.py:652] Step 79400 per-step time 0.256s loss=0.869
    INFO:tensorflow:Step 79500 per-step time 0.255s loss=1.335
    I1009 11:23:11.292413 139781364750208 model_lib_v2.py:652] Step 79500 per-step time 0.255s loss=1.335
    INFO:tensorflow:Step 79600 per-step time 0.257s loss=0.728
    I1009 11:23:35.950632 139781364750208 model_lib_v2.py:652] Step 79600 per-step time 0.257s loss=0.728
    INFO:tensorflow:Step 79700 per-step time 0.253s loss=0.667
    I1009 11:24:00.454761 139781364750208 model_lib_v2.py:652] Step 79700 per-step time 0.253s loss=0.667
    INFO:tensorflow:Step 79800 per-step time 0.242s loss=1.304
    I1009 11:24:25.227358 139781364750208 model_lib_v2.py:652] Step 79800 per-step time 0.242s loss=1.304
    INFO:tensorflow:Step 79900 per-step time 0.250s loss=0.740
    I1009 11:24:50.064805 139781364750208 model_lib_v2.py:652] Step 79900 per-step time 0.250s loss=0.740
    INFO:tensorflow:Step 80000 per-step time 0.241s loss=1.121
    I1009 11:25:14.842437 139781364750208 model_lib_v2.py:652] Step 80000 per-step time 0.241s loss=1.121
    INFO:tensorflow:Step 80100 per-step time 0.243s loss=0.808
    I1009 11:25:40.584302 139781364750208 model_lib_v2.py:652] Step 80100 per-step time 0.243s loss=0.808
    INFO:tensorflow:Step 80200 per-step time 0.243s loss=1.117
    I1009 11:26:05.539038 139781364750208 model_lib_v2.py:652] Step 80200 per-step time 0.243s loss=1.117
    INFO:tensorflow:Step 80300 per-step time 0.233s loss=0.740
    I1009 11:26:30.400996 139781364750208 model_lib_v2.py:652] Step 80300 per-step time 0.233s loss=0.740
    INFO:tensorflow:Step 80400 per-step time 0.240s loss=0.928
    I1009 11:26:55.077754 139781364750208 model_lib_v2.py:652] Step 80400 per-step time 0.240s loss=0.928
    INFO:tensorflow:Step 80500 per-step time 0.245s loss=0.864
    I1009 11:27:19.833971 139781364750208 model_lib_v2.py:652] Step 80500 per-step time 0.245s loss=0.864
    INFO:tensorflow:Step 80600 per-step time 0.232s loss=1.003
    I1009 11:27:44.569663 139781364750208 model_lib_v2.py:652] Step 80600 per-step time 0.232s loss=1.003
    INFO:tensorflow:Step 80700 per-step time 0.234s loss=1.035
    I1009 11:28:09.447137 139781364750208 model_lib_v2.py:652] Step 80700 per-step time 0.234s loss=1.035
    INFO:tensorflow:Step 80800 per-step time 0.248s loss=1.282
    I1009 11:28:34.016400 139781364750208 model_lib_v2.py:652] Step 80800 per-step time 0.248s loss=1.282
    INFO:tensorflow:Step 80900 per-step time 0.247s loss=0.904
    I1009 11:28:58.764543 139781364750208 model_lib_v2.py:652] Step 80900 per-step time 0.247s loss=0.904
    INFO:tensorflow:Step 81000 per-step time 0.243s loss=0.844
    I1009 11:29:23.608421 139781364750208 model_lib_v2.py:652] Step 81000 per-step time 0.243s loss=0.844
    INFO:tensorflow:Step 81100 per-step time 0.239s loss=0.909
    I1009 11:29:49.221986 139781364750208 model_lib_v2.py:652] Step 81100 per-step time 0.239s loss=0.909
    INFO:tensorflow:Step 81200 per-step time 0.259s loss=0.896
    I1009 11:30:14.076205 139781364750208 model_lib_v2.py:652] Step 81200 per-step time 0.259s loss=0.896
    INFO:tensorflow:Step 81300 per-step time 0.243s loss=0.933
    I1009 11:30:38.838937 139781364750208 model_lib_v2.py:652] Step 81300 per-step time 0.243s loss=0.933
    INFO:tensorflow:Step 81400 per-step time 0.249s loss=0.877
    I1009 11:31:03.426580 139781364750208 model_lib_v2.py:652] Step 81400 per-step time 0.249s loss=0.877
    INFO:tensorflow:Step 81500 per-step time 0.235s loss=0.693
    I1009 11:31:28.150555 139781364750208 model_lib_v2.py:652] Step 81500 per-step time 0.235s loss=0.693
    INFO:tensorflow:Step 81600 per-step time 0.265s loss=0.555
    I1009 11:31:52.690399 139781364750208 model_lib_v2.py:652] Step 81600 per-step time 0.265s loss=0.555
    INFO:tensorflow:Step 81700 per-step time 0.237s loss=0.907
    I1009 11:32:17.148430 139781364750208 model_lib_v2.py:652] Step 81700 per-step time 0.237s loss=0.907
    INFO:tensorflow:Step 81800 per-step time 0.248s loss=0.966
    I1009 11:32:41.799375 139781364750208 model_lib_v2.py:652] Step 81800 per-step time 0.248s loss=0.966
    INFO:tensorflow:Step 81900 per-step time 0.254s loss=0.582
    I1009 11:33:06.452147 139781364750208 model_lib_v2.py:652] Step 81900 per-step time 0.254s loss=0.582
    INFO:tensorflow:Step 82000 per-step time 0.248s loss=0.674
    I1009 11:33:31.169750 139781364750208 model_lib_v2.py:652] Step 82000 per-step time 0.248s loss=0.674
    INFO:tensorflow:Step 82100 per-step time 0.240s loss=0.601
    I1009 11:33:56.880226 139781364750208 model_lib_v2.py:652] Step 82100 per-step time 0.240s loss=0.601
    INFO:tensorflow:Step 82200 per-step time 0.239s loss=1.042
    I1009 11:34:21.394064 139781364750208 model_lib_v2.py:652] Step 82200 per-step time 0.239s loss=1.042
    INFO:tensorflow:Step 82300 per-step time 0.241s loss=0.666
    I1009 11:34:46.053598 139781364750208 model_lib_v2.py:652] Step 82300 per-step time 0.241s loss=0.666
    INFO:tensorflow:Step 82400 per-step time 0.245s loss=1.001
    I1009 11:35:10.727019 139781364750208 model_lib_v2.py:652] Step 82400 per-step time 0.245s loss=1.001
    INFO:tensorflow:Step 82500 per-step time 0.245s loss=0.885
    I1009 11:35:35.390734 139781364750208 model_lib_v2.py:652] Step 82500 per-step time 0.245s loss=0.885
    INFO:tensorflow:Step 82600 per-step time 0.250s loss=0.672
    I1009 11:35:59.922572 139781364750208 model_lib_v2.py:652] Step 82600 per-step time 0.250s loss=0.672
    INFO:tensorflow:Step 82700 per-step time 0.239s loss=0.926
    I1009 11:36:24.538926 139781364750208 model_lib_v2.py:652] Step 82700 per-step time 0.239s loss=0.926
    INFO:tensorflow:Step 82800 per-step time 0.256s loss=0.762
    I1009 11:36:49.182381 139781364750208 model_lib_v2.py:652] Step 82800 per-step time 0.256s loss=0.762
    INFO:tensorflow:Step 82900 per-step time 0.250s loss=0.863
    I1009 11:37:14.014819 139781364750208 model_lib_v2.py:652] Step 82900 per-step time 0.250s loss=0.863
    INFO:tensorflow:Step 83000 per-step time 0.259s loss=0.768
    I1009 11:37:38.904227 139781364750208 model_lib_v2.py:652] Step 83000 per-step time 0.259s loss=0.768
    INFO:tensorflow:Step 83100 per-step time 0.233s loss=0.829
    I1009 11:38:04.792623 139781364750208 model_lib_v2.py:652] Step 83100 per-step time 0.233s loss=0.829
    INFO:tensorflow:Step 83200 per-step time 0.243s loss=0.801
    I1009 11:38:29.794598 139781364750208 model_lib_v2.py:652] Step 83200 per-step time 0.243s loss=0.801
    INFO:tensorflow:Step 83300 per-step time 0.245s loss=1.081
    I1009 11:38:54.309017 139781364750208 model_lib_v2.py:652] Step 83300 per-step time 0.245s loss=1.081
    INFO:tensorflow:Step 83400 per-step time 0.246s loss=0.882
    I1009 11:39:18.980467 139781364750208 model_lib_v2.py:652] Step 83400 per-step time 0.246s loss=0.882
    INFO:tensorflow:Step 83500 per-step time 0.257s loss=0.675
    I1009 11:39:43.596806 139781364750208 model_lib_v2.py:652] Step 83500 per-step time 0.257s loss=0.675
    INFO:tensorflow:Step 83600 per-step time 0.237s loss=0.861
    I1009 11:40:08.253263 139781364750208 model_lib_v2.py:652] Step 83600 per-step time 0.237s loss=0.861
    INFO:tensorflow:Step 83700 per-step time 0.252s loss=0.827
    I1009 11:40:33.025882 139781364750208 model_lib_v2.py:652] Step 83700 per-step time 0.252s loss=0.827
    INFO:tensorflow:Step 83800 per-step time 0.252s loss=0.629
    I1009 11:40:57.663336 139781364750208 model_lib_v2.py:652] Step 83800 per-step time 0.252s loss=0.629
    INFO:tensorflow:Step 83900 per-step time 0.244s loss=0.694
    I1009 11:41:22.286521 139781364750208 model_lib_v2.py:652] Step 83900 per-step time 0.244s loss=0.694
    INFO:tensorflow:Step 84000 per-step time 0.262s loss=0.910
    I1009 11:41:46.803920 139781364750208 model_lib_v2.py:652] Step 84000 per-step time 0.262s loss=0.910
    INFO:tensorflow:Step 84100 per-step time 0.253s loss=0.983
    I1009 11:42:12.148133 139781364750208 model_lib_v2.py:652] Step 84100 per-step time 0.253s loss=0.983
    INFO:tensorflow:Step 84200 per-step time 0.253s loss=0.885
    I1009 11:42:36.564294 139781364750208 model_lib_v2.py:652] Step 84200 per-step time 0.253s loss=0.885
    INFO:tensorflow:Step 84300 per-step time 0.238s loss=0.869
    I1009 11:43:01.048394 139781364750208 model_lib_v2.py:652] Step 84300 per-step time 0.238s loss=0.869
    INFO:tensorflow:Step 84400 per-step time 0.239s loss=0.823
    I1009 11:43:25.597321 139781364750208 model_lib_v2.py:652] Step 84400 per-step time 0.239s loss=0.823
    INFO:tensorflow:Step 84500 per-step time 0.251s loss=0.549
    I1009 11:43:50.378873 139781364750208 model_lib_v2.py:652] Step 84500 per-step time 0.251s loss=0.549
    INFO:tensorflow:Step 84600 per-step time 0.242s loss=0.717
    I1009 11:44:14.864336 139781364750208 model_lib_v2.py:652] Step 84600 per-step time 0.242s loss=0.717
    INFO:tensorflow:Step 84700 per-step time 0.240s loss=0.538
    I1009 11:44:39.301797 139781364750208 model_lib_v2.py:652] Step 84700 per-step time 0.240s loss=0.538
    INFO:tensorflow:Step 84800 per-step time 0.238s loss=0.985
    I1009 11:45:03.823184 139781364750208 model_lib_v2.py:652] Step 84800 per-step time 0.238s loss=0.985
    INFO:tensorflow:Step 84900 per-step time 0.241s loss=0.867
    I1009 11:45:28.498030 139781364750208 model_lib_v2.py:652] Step 84900 per-step time 0.241s loss=0.867
    INFO:tensorflow:Step 85000 per-step time 0.240s loss=0.476
    I1009 11:45:53.143019 139781364750208 model_lib_v2.py:652] Step 85000 per-step time 0.240s loss=0.476
    INFO:tensorflow:Step 85100 per-step time 0.233s loss=1.045
    I1009 11:46:18.659975 139781364750208 model_lib_v2.py:652] Step 85100 per-step time 0.233s loss=1.045
    INFO:tensorflow:Step 85200 per-step time 0.239s loss=0.542
    I1009 11:46:43.359659 139781364750208 model_lib_v2.py:652] Step 85200 per-step time 0.239s loss=0.542
    INFO:tensorflow:Step 85300 per-step time 0.251s loss=0.685
    I1009 11:47:07.939819 139781364750208 model_lib_v2.py:652] Step 85300 per-step time 0.251s loss=0.685
    INFO:tensorflow:Step 85400 per-step time 0.248s loss=0.688
    I1009 11:47:32.428038 139781364750208 model_lib_v2.py:652] Step 85400 per-step time 0.248s loss=0.688
    INFO:tensorflow:Step 85500 per-step time 0.239s loss=0.720
    I1009 11:47:57.013260 139781364750208 model_lib_v2.py:652] Step 85500 per-step time 0.239s loss=0.720
    INFO:tensorflow:Step 85600 per-step time 0.240s loss=0.694
    I1009 11:48:21.704476 139781364750208 model_lib_v2.py:652] Step 85600 per-step time 0.240s loss=0.694
    INFO:tensorflow:Step 85700 per-step time 0.244s loss=0.631
    I1009 11:48:46.421096 139781364750208 model_lib_v2.py:652] Step 85700 per-step time 0.244s loss=0.631
    INFO:tensorflow:Step 85800 per-step time 0.252s loss=0.691
    I1009 11:49:10.924193 139781364750208 model_lib_v2.py:652] Step 85800 per-step time 0.252s loss=0.691
    INFO:tensorflow:Step 85900 per-step time 0.254s loss=0.609
    I1009 11:49:35.733206 139781364750208 model_lib_v2.py:652] Step 85900 per-step time 0.254s loss=0.609
    INFO:tensorflow:Step 86000 per-step time 0.249s loss=0.873
    I1009 11:50:00.528980 139781364750208 model_lib_v2.py:652] Step 86000 per-step time 0.249s loss=0.873
    INFO:tensorflow:Step 86100 per-step time 0.241s loss=0.781
    I1009 11:50:26.037326 139781364750208 model_lib_v2.py:652] Step 86100 per-step time 0.241s loss=0.781
    INFO:tensorflow:Step 86200 per-step time 0.235s loss=1.124
    I1009 11:50:50.968091 139781364750208 model_lib_v2.py:652] Step 86200 per-step time 0.235s loss=1.124
    INFO:tensorflow:Step 86300 per-step time 0.244s loss=1.018
    I1009 11:51:15.650984 139781364750208 model_lib_v2.py:652] Step 86300 per-step time 0.244s loss=1.018
    INFO:tensorflow:Step 86400 per-step time 0.250s loss=0.908
    I1009 11:51:40.246442 139781364750208 model_lib_v2.py:652] Step 86400 per-step time 0.250s loss=0.908
    INFO:tensorflow:Step 86500 per-step time 0.245s loss=0.680
    I1009 11:52:04.876553 139781364750208 model_lib_v2.py:652] Step 86500 per-step time 0.245s loss=0.680
    INFO:tensorflow:Step 86600 per-step time 0.241s loss=0.695
    I1009 11:52:29.446989 139781364750208 model_lib_v2.py:652] Step 86600 per-step time 0.241s loss=0.695
    INFO:tensorflow:Step 86700 per-step time 0.248s loss=0.799
    I1009 11:52:53.995535 139781364750208 model_lib_v2.py:652] Step 86700 per-step time 0.248s loss=0.799
    INFO:tensorflow:Step 86800 per-step time 0.247s loss=0.786
    I1009 11:53:18.575483 139781364750208 model_lib_v2.py:652] Step 86800 per-step time 0.247s loss=0.786
    INFO:tensorflow:Step 86900 per-step time 0.255s loss=0.739
    I1009 11:53:43.333222 139781364750208 model_lib_v2.py:652] Step 86900 per-step time 0.255s loss=0.739
    INFO:tensorflow:Step 87000 per-step time 0.243s loss=0.834
    I1009 11:54:08.222023 139781364750208 model_lib_v2.py:652] Step 87000 per-step time 0.243s loss=0.834
    INFO:tensorflow:Step 87100 per-step time 0.247s loss=0.803
    I1009 11:54:33.817303 139781364750208 model_lib_v2.py:652] Step 87100 per-step time 0.247s loss=0.803
    INFO:tensorflow:Step 87200 per-step time 0.255s loss=0.824
    I1009 11:54:58.626588 139781364750208 model_lib_v2.py:652] Step 87200 per-step time 0.255s loss=0.824
    INFO:tensorflow:Step 87300 per-step time 0.259s loss=1.216
    I1009 11:55:23.438711 139781364750208 model_lib_v2.py:652] Step 87300 per-step time 0.259s loss=1.216
    INFO:tensorflow:Step 87400 per-step time 0.245s loss=0.896
    I1009 11:55:48.306035 139781364750208 model_lib_v2.py:652] Step 87400 per-step time 0.245s loss=0.896
    INFO:tensorflow:Step 87500 per-step time 0.251s loss=0.714
    I1009 11:56:13.289028 139781364750208 model_lib_v2.py:652] Step 87500 per-step time 0.251s loss=0.714
    INFO:tensorflow:Step 87600 per-step time 0.248s loss=0.778
    I1009 11:56:38.403981 139781364750208 model_lib_v2.py:652] Step 87600 per-step time 0.248s loss=0.778
    INFO:tensorflow:Step 87700 per-step time 0.252s loss=0.788
    I1009 11:57:03.451383 139781364750208 model_lib_v2.py:652] Step 87700 per-step time 0.252s loss=0.788
    INFO:tensorflow:Step 87800 per-step time 0.236s loss=0.888
    I1009 11:57:28.423864 139781364750208 model_lib_v2.py:652] Step 87800 per-step time 0.236s loss=0.888
    INFO:tensorflow:Step 87900 per-step time 0.248s loss=0.853
    I1009 11:57:53.505110 139781364750208 model_lib_v2.py:652] Step 87900 per-step time 0.248s loss=0.853
    INFO:tensorflow:Step 88000 per-step time 0.254s loss=0.546
    I1009 11:58:18.478755 139781364750208 model_lib_v2.py:652] Step 88000 per-step time 0.254s loss=0.546
    INFO:tensorflow:Step 88100 per-step time 0.256s loss=1.007
    I1009 11:58:44.473076 139781364750208 model_lib_v2.py:652] Step 88100 per-step time 0.256s loss=1.007
    INFO:tensorflow:Step 88200 per-step time 0.245s loss=0.605
    I1009 11:59:09.613685 139781364750208 model_lib_v2.py:652] Step 88200 per-step time 0.245s loss=0.605
    INFO:tensorflow:Step 88300 per-step time 0.246s loss=0.728
    I1009 11:59:34.761992 139781364750208 model_lib_v2.py:652] Step 88300 per-step time 0.246s loss=0.728
    INFO:tensorflow:Step 88400 per-step time 0.247s loss=0.904
    I1009 11:59:59.753601 139781364750208 model_lib_v2.py:652] Step 88400 per-step time 0.247s loss=0.904
    INFO:tensorflow:Step 88500 per-step time 0.239s loss=0.917
    I1009 12:00:24.719216 139781364750208 model_lib_v2.py:652] Step 88500 per-step time 0.239s loss=0.917
    INFO:tensorflow:Step 88600 per-step time 0.243s loss=0.628
    I1009 12:00:49.566386 139781364750208 model_lib_v2.py:652] Step 88600 per-step time 0.243s loss=0.628
    INFO:tensorflow:Step 88700 per-step time 0.249s loss=0.785
    I1009 12:01:14.543032 139781364750208 model_lib_v2.py:652] Step 88700 per-step time 0.249s loss=0.785
    INFO:tensorflow:Step 88800 per-step time 0.249s loss=1.082
    I1009 12:01:39.577075 139781364750208 model_lib_v2.py:652] Step 88800 per-step time 0.249s loss=1.082
    INFO:tensorflow:Step 88900 per-step time 0.259s loss=0.875
    I1009 12:02:04.572080 139781364750208 model_lib_v2.py:652] Step 88900 per-step time 0.259s loss=0.875
    INFO:tensorflow:Step 89000 per-step time 0.247s loss=0.613
    I1009 12:02:29.503995 139781364750208 model_lib_v2.py:652] Step 89000 per-step time 0.247s loss=0.613
    INFO:tensorflow:Step 89100 per-step time 0.248s loss=0.830
    I1009 12:02:55.222990 139781364750208 model_lib_v2.py:652] Step 89100 per-step time 0.248s loss=0.830
    INFO:tensorflow:Step 89200 per-step time 0.248s loss=0.816
    I1009 12:03:20.248129 139781364750208 model_lib_v2.py:652] Step 89200 per-step time 0.248s loss=0.816
    INFO:tensorflow:Step 89300 per-step time 0.264s loss=0.716
    I1009 12:03:45.286212 139781364750208 model_lib_v2.py:652] Step 89300 per-step time 0.264s loss=0.716
    INFO:tensorflow:Step 89400 per-step time 0.255s loss=0.692
    I1009 12:04:10.327588 139781364750208 model_lib_v2.py:652] Step 89400 per-step time 0.255s loss=0.692
    INFO:tensorflow:Step 89500 per-step time 0.253s loss=0.792
    I1009 12:04:35.652789 139781364750208 model_lib_v2.py:652] Step 89500 per-step time 0.253s loss=0.792
    INFO:tensorflow:Step 89600 per-step time 0.244s loss=0.793
    I1009 12:05:00.696899 139781364750208 model_lib_v2.py:652] Step 89600 per-step time 0.244s loss=0.793
    INFO:tensorflow:Step 89700 per-step time 0.246s loss=0.776
    I1009 12:05:25.507979 139781364750208 model_lib_v2.py:652] Step 89700 per-step time 0.246s loss=0.776
    INFO:tensorflow:Step 89800 per-step time 0.251s loss=0.694
    I1009 12:05:50.477381 139781364750208 model_lib_v2.py:652] Step 89800 per-step time 0.251s loss=0.694
    INFO:tensorflow:Step 89900 per-step time 0.243s loss=0.861
    I1009 12:06:15.453541 139781364750208 model_lib_v2.py:652] Step 89900 per-step time 0.243s loss=0.861
    INFO:tensorflow:Step 90000 per-step time 0.246s loss=0.880
    I1009 12:06:40.212836 139781364750208 model_lib_v2.py:652] Step 90000 per-step time 0.246s loss=0.880
    INFO:tensorflow:Step 90100 per-step time 0.246s loss=0.902
    I1009 12:07:05.961110 139781364750208 model_lib_v2.py:652] Step 90100 per-step time 0.246s loss=0.902
    INFO:tensorflow:Step 90200 per-step time 0.247s loss=0.993
    I1009 12:07:30.848461 139781364750208 model_lib_v2.py:652] Step 90200 per-step time 0.247s loss=0.993
    INFO:tensorflow:Step 90300 per-step time 0.257s loss=0.712
    I1009 12:07:56.247632 139781364750208 model_lib_v2.py:652] Step 90300 per-step time 0.257s loss=0.712
    INFO:tensorflow:Step 90400 per-step time 0.250s loss=0.986
    I1009 12:08:21.330335 139781364750208 model_lib_v2.py:652] Step 90400 per-step time 0.250s loss=0.986
    INFO:tensorflow:Step 90500 per-step time 0.266s loss=1.120
    I1009 12:08:46.349656 139781364750208 model_lib_v2.py:652] Step 90500 per-step time 0.266s loss=1.120
    INFO:tensorflow:Step 90600 per-step time 0.246s loss=0.667
    I1009 12:09:11.255453 139781364750208 model_lib_v2.py:652] Step 90600 per-step time 0.246s loss=0.667
    INFO:tensorflow:Step 90700 per-step time 0.249s loss=0.943
    I1009 12:09:36.419095 139781364750208 model_lib_v2.py:652] Step 90700 per-step time 0.249s loss=0.943
    INFO:tensorflow:Step 90800 per-step time 0.258s loss=0.805
    I1009 12:10:01.350294 139781364750208 model_lib_v2.py:652] Step 90800 per-step time 0.258s loss=0.805
    INFO:tensorflow:Step 90900 per-step time 0.250s loss=0.868
    I1009 12:10:26.364014 139781364750208 model_lib_v2.py:652] Step 90900 per-step time 0.250s loss=0.868
    INFO:tensorflow:Step 91000 per-step time 0.245s loss=0.524
    I1009 12:10:51.378413 139781364750208 model_lib_v2.py:652] Step 91000 per-step time 0.245s loss=0.524
    INFO:tensorflow:Step 91100 per-step time 0.249s loss=0.805
    I1009 12:11:17.413735 139781364750208 model_lib_v2.py:652] Step 91100 per-step time 0.249s loss=0.805
    INFO:tensorflow:Step 91200 per-step time 0.254s loss=0.519
    I1009 12:11:42.319595 139781364750208 model_lib_v2.py:652] Step 91200 per-step time 0.254s loss=0.519
    INFO:tensorflow:Step 91300 per-step time 0.266s loss=0.651
    I1009 12:12:07.267210 139781364750208 model_lib_v2.py:652] Step 91300 per-step time 0.266s loss=0.651
    INFO:tensorflow:Step 91400 per-step time 0.256s loss=0.929
    I1009 12:12:32.156048 139781364750208 model_lib_v2.py:652] Step 91400 per-step time 0.256s loss=0.929
    INFO:tensorflow:Step 91500 per-step time 0.245s loss=0.592
    I1009 12:12:57.171151 139781364750208 model_lib_v2.py:652] Step 91500 per-step time 0.245s loss=0.592
    INFO:tensorflow:Step 91600 per-step time 0.235s loss=0.682
    I1009 12:13:21.951967 139781364750208 model_lib_v2.py:652] Step 91600 per-step time 0.235s loss=0.682
    INFO:tensorflow:Step 91700 per-step time 0.255s loss=0.514
    I1009 12:13:46.889116 139781364750208 model_lib_v2.py:652] Step 91700 per-step time 0.255s loss=0.514
    INFO:tensorflow:Step 91800 per-step time 0.248s loss=0.841
    I1009 12:14:11.795419 139781364750208 model_lib_v2.py:652] Step 91800 per-step time 0.248s loss=0.841
    INFO:tensorflow:Step 91900 per-step time 0.255s loss=0.746
    I1009 12:14:36.965187 139781364750208 model_lib_v2.py:652] Step 91900 per-step time 0.255s loss=0.746
    INFO:tensorflow:Step 92000 per-step time 0.256s loss=0.902
    I1009 12:15:02.142049 139781364750208 model_lib_v2.py:652] Step 92000 per-step time 0.256s loss=0.902
    INFO:tensorflow:Step 92100 per-step time 0.260s loss=1.126
    I1009 12:15:28.045362 139781364750208 model_lib_v2.py:652] Step 92100 per-step time 0.260s loss=1.126
    INFO:tensorflow:Step 92200 per-step time 0.247s loss=0.670
    I1009 12:15:53.115665 139781364750208 model_lib_v2.py:652] Step 92200 per-step time 0.247s loss=0.670
    INFO:tensorflow:Step 92300 per-step time 0.247s loss=1.044
    I1009 12:16:18.253087 139781364750208 model_lib_v2.py:652] Step 92300 per-step time 0.247s loss=1.044
    INFO:tensorflow:Step 92400 per-step time 0.258s loss=0.657
    I1009 12:16:43.204174 139781364750208 model_lib_v2.py:652] Step 92400 per-step time 0.258s loss=0.657
    INFO:tensorflow:Step 92500 per-step time 0.250s loss=0.623
    I1009 12:17:08.189208 139781364750208 model_lib_v2.py:652] Step 92500 per-step time 0.250s loss=0.623
    INFO:tensorflow:Step 92600 per-step time 0.249s loss=0.755
    I1009 12:17:33.043886 139781364750208 model_lib_v2.py:652] Step 92600 per-step time 0.249s loss=0.755
    INFO:tensorflow:Step 92700 per-step time 0.243s loss=0.787
    I1009 12:17:57.974003 139781364750208 model_lib_v2.py:652] Step 92700 per-step time 0.243s loss=0.787
    INFO:tensorflow:Step 92800 per-step time 0.241s loss=0.743
    I1009 12:18:22.789765 139781364750208 model_lib_v2.py:652] Step 92800 per-step time 0.241s loss=0.743
    INFO:tensorflow:Step 92900 per-step time 0.246s loss=0.890
    I1009 12:18:47.484993 139781364750208 model_lib_v2.py:652] Step 92900 per-step time 0.246s loss=0.890
    INFO:tensorflow:Step 93000 per-step time 0.256s loss=0.764
    I1009 12:19:12.124941 139781364750208 model_lib_v2.py:652] Step 93000 per-step time 0.256s loss=0.764
    INFO:tensorflow:Step 93100 per-step time 0.241s loss=0.816
    I1009 12:19:37.569041 139781364750208 model_lib_v2.py:652] Step 93100 per-step time 0.241s loss=0.816
    INFO:tensorflow:Step 93200 per-step time 0.244s loss=0.804
    I1009 12:20:02.628251 139781364750208 model_lib_v2.py:652] Step 93200 per-step time 0.244s loss=0.804
    INFO:tensorflow:Step 93300 per-step time 0.250s loss=0.562
    I1009 12:20:27.318033 139781364750208 model_lib_v2.py:652] Step 93300 per-step time 0.250s loss=0.562
    INFO:tensorflow:Step 93400 per-step time 0.249s loss=1.073
    I1009 12:20:52.067235 139781364750208 model_lib_v2.py:652] Step 93400 per-step time 0.249s loss=1.073
    INFO:tensorflow:Step 93500 per-step time 0.240s loss=0.539
    I1009 12:21:16.927742 139781364750208 model_lib_v2.py:652] Step 93500 per-step time 0.240s loss=0.539
    INFO:tensorflow:Step 93600 per-step time 0.244s loss=0.691
    I1009 12:21:41.706162 139781364750208 model_lib_v2.py:652] Step 93600 per-step time 0.244s loss=0.691
    INFO:tensorflow:Step 93700 per-step time 0.254s loss=0.730
    I1009 12:22:06.452046 139781364750208 model_lib_v2.py:652] Step 93700 per-step time 0.254s loss=0.730
    INFO:tensorflow:Step 93800 per-step time 0.254s loss=0.810
    I1009 12:22:31.395433 139781364750208 model_lib_v2.py:652] Step 93800 per-step time 0.254s loss=0.810
    INFO:tensorflow:Step 93900 per-step time 0.247s loss=1.121
    I1009 12:22:56.231690 139781364750208 model_lib_v2.py:652] Step 93900 per-step time 0.247s loss=1.121
    INFO:tensorflow:Step 94000 per-step time 0.243s loss=1.020
    I1009 12:23:20.937317 139781364750208 model_lib_v2.py:652] Step 94000 per-step time 0.243s loss=1.020
    INFO:tensorflow:Step 94100 per-step time 0.246s loss=0.889
    I1009 12:23:47.042992 139781364750208 model_lib_v2.py:652] Step 94100 per-step time 0.246s loss=0.889
    INFO:tensorflow:Step 94200 per-step time 0.250s loss=1.192
    I1009 12:24:11.653089 139781364750208 model_lib_v2.py:652] Step 94200 per-step time 0.250s loss=1.192
    INFO:tensorflow:Step 94300 per-step time 0.246s loss=0.785
    I1009 12:24:36.424442 139781364750208 model_lib_v2.py:652] Step 94300 per-step time 0.246s loss=0.785
    INFO:tensorflow:Step 94400 per-step time 0.259s loss=0.811
    I1009 12:25:01.465975 139781364750208 model_lib_v2.py:652] Step 94400 per-step time 0.259s loss=0.811
    INFO:tensorflow:Step 94500 per-step time 0.258s loss=0.834
    I1009 12:25:26.348713 139781364750208 model_lib_v2.py:652] Step 94500 per-step time 0.258s loss=0.834
    INFO:tensorflow:Step 94600 per-step time 0.241s loss=0.690
    I1009 12:25:51.018615 139781364750208 model_lib_v2.py:652] Step 94600 per-step time 0.241s loss=0.690
    INFO:tensorflow:Step 94700 per-step time 0.250s loss=1.007
    I1009 12:26:15.736108 139781364750208 model_lib_v2.py:652] Step 94700 per-step time 0.250s loss=1.007
    INFO:tensorflow:Step 94800 per-step time 0.247s loss=1.076
    I1009 12:26:40.438082 139781364750208 model_lib_v2.py:652] Step 94800 per-step time 0.247s loss=1.076
    INFO:tensorflow:Step 94900 per-step time 0.241s loss=0.756
    I1009 12:27:05.302443 139781364750208 model_lib_v2.py:652] Step 94900 per-step time 0.241s loss=0.756
    INFO:tensorflow:Step 95000 per-step time 0.236s loss=0.814
    I1009 12:27:30.067284 139781364750208 model_lib_v2.py:652] Step 95000 per-step time 0.236s loss=0.814
    INFO:tensorflow:Step 95100 per-step time 0.237s loss=0.429
    I1009 12:27:55.812718 139781364750208 model_lib_v2.py:652] Step 95100 per-step time 0.237s loss=0.429
    INFO:tensorflow:Step 95200 per-step time 0.269s loss=0.707
    I1009 12:28:20.341892 139781364750208 model_lib_v2.py:652] Step 95200 per-step time 0.269s loss=0.707
    INFO:tensorflow:Step 95300 per-step time 0.255s loss=0.746
    I1009 12:28:45.014131 139781364750208 model_lib_v2.py:652] Step 95300 per-step time 0.255s loss=0.746
    INFO:tensorflow:Step 95400 per-step time 0.245s loss=0.494
    I1009 12:29:09.852247 139781364750208 model_lib_v2.py:652] Step 95400 per-step time 0.245s loss=0.494
    INFO:tensorflow:Step 95500 per-step time 0.239s loss=1.250
    I1009 12:29:34.634149 139781364750208 model_lib_v2.py:652] Step 95500 per-step time 0.239s loss=1.250
    INFO:tensorflow:Step 95600 per-step time 0.256s loss=0.931
    I1009 12:29:59.415413 139781364750208 model_lib_v2.py:652] Step 95600 per-step time 0.256s loss=0.931
    INFO:tensorflow:Step 95700 per-step time 0.253s loss=0.733
    I1009 12:30:24.498452 139781364750208 model_lib_v2.py:652] Step 95700 per-step time 0.253s loss=0.733
    INFO:tensorflow:Step 95800 per-step time 0.238s loss=0.654
    I1009 12:30:49.160424 139781364750208 model_lib_v2.py:652] Step 95800 per-step time 0.238s loss=0.654
    INFO:tensorflow:Step 95900 per-step time 0.244s loss=0.950
    I1009 12:31:13.799610 139781364750208 model_lib_v2.py:652] Step 95900 per-step time 0.244s loss=0.950
    INFO:tensorflow:Step 96000 per-step time 0.237s loss=0.886
    I1009 12:31:38.580712 139781364750208 model_lib_v2.py:652] Step 96000 per-step time 0.237s loss=0.886
    INFO:tensorflow:Step 96100 per-step time 0.255s loss=0.596
    I1009 12:32:04.140438 139781364750208 model_lib_v2.py:652] Step 96100 per-step time 0.255s loss=0.596
    INFO:tensorflow:Step 96200 per-step time 0.240s loss=1.254
    I1009 12:32:28.738751 139781364750208 model_lib_v2.py:652] Step 96200 per-step time 0.240s loss=1.254
    INFO:tensorflow:Step 96300 per-step time 0.246s loss=0.765
    I1009 12:32:53.315484 139781364750208 model_lib_v2.py:652] Step 96300 per-step time 0.246s loss=0.765
    INFO:tensorflow:Step 96400 per-step time 0.248s loss=0.881
    I1009 12:33:17.987304 139781364750208 model_lib_v2.py:652] Step 96400 per-step time 0.248s loss=0.881
    INFO:tensorflow:Step 96500 per-step time 0.243s loss=0.737
    I1009 12:33:42.614527 139781364750208 model_lib_v2.py:652] Step 96500 per-step time 0.243s loss=0.737
    INFO:tensorflow:Step 96600 per-step time 0.237s loss=0.688
    I1009 12:34:07.216666 139781364750208 model_lib_v2.py:652] Step 96600 per-step time 0.237s loss=0.688
    INFO:tensorflow:Step 96700 per-step time 0.244s loss=0.436
    I1009 12:34:31.571928 139781364750208 model_lib_v2.py:652] Step 96700 per-step time 0.244s loss=0.436
    INFO:tensorflow:Step 96800 per-step time 0.239s loss=0.680
    I1009 12:34:55.851621 139781364750208 model_lib_v2.py:652] Step 96800 per-step time 0.239s loss=0.680
    INFO:tensorflow:Step 96900 per-step time 0.256s loss=0.889
    I1009 12:35:20.593796 139781364750208 model_lib_v2.py:652] Step 96900 per-step time 0.256s loss=0.889
    INFO:tensorflow:Step 97000 per-step time 0.258s loss=0.825
    I1009 12:35:45.176922 139781364750208 model_lib_v2.py:652] Step 97000 per-step time 0.258s loss=0.825
    INFO:tensorflow:Step 97100 per-step time 0.242s loss=0.752
    I1009 12:36:10.697295 139781364750208 model_lib_v2.py:652] Step 97100 per-step time 0.242s loss=0.752
    INFO:tensorflow:Step 97200 per-step time 0.237s loss=0.715
    I1009 12:36:35.219150 139781364750208 model_lib_v2.py:652] Step 97200 per-step time 0.237s loss=0.715
    INFO:tensorflow:Step 97300 per-step time 0.251s loss=0.684
    I1009 12:36:59.787682 139781364750208 model_lib_v2.py:652] Step 97300 per-step time 0.251s loss=0.684
    INFO:tensorflow:Step 97400 per-step time 0.252s loss=0.946
    I1009 12:37:24.487312 139781364750208 model_lib_v2.py:652] Step 97400 per-step time 0.252s loss=0.946
    INFO:tensorflow:Step 97500 per-step time 0.254s loss=0.894
    I1009 12:37:48.892091 139781364750208 model_lib_v2.py:652] Step 97500 per-step time 0.254s loss=0.894
    INFO:tensorflow:Step 97600 per-step time 0.250s loss=0.777
    I1009 12:38:13.617113 139781364750208 model_lib_v2.py:652] Step 97600 per-step time 0.250s loss=0.777
    INFO:tensorflow:Step 97700 per-step time 0.252s loss=0.687
    I1009 12:38:37.981642 139781364750208 model_lib_v2.py:652] Step 97700 per-step time 0.252s loss=0.687
    INFO:tensorflow:Step 97800 per-step time 0.246s loss=0.987
    I1009 12:39:02.511331 139781364750208 model_lib_v2.py:652] Step 97800 per-step time 0.246s loss=0.987
    INFO:tensorflow:Step 97900 per-step time 0.259s loss=0.631
    I1009 12:39:27.001069 139781364750208 model_lib_v2.py:652] Step 97900 per-step time 0.259s loss=0.631
    INFO:tensorflow:Step 98000 per-step time 0.249s loss=0.626
    I1009 12:39:51.485876 139781364750208 model_lib_v2.py:652] Step 98000 per-step time 0.249s loss=0.626
    INFO:tensorflow:Step 98100 per-step time 0.254s loss=1.052
    I1009 12:40:17.427846 139781364750208 model_lib_v2.py:652] Step 98100 per-step time 0.254s loss=1.052
    INFO:tensorflow:Step 98200 per-step time 0.248s loss=0.715
    I1009 12:40:42.266750 139781364750208 model_lib_v2.py:652] Step 98200 per-step time 0.248s loss=0.715
    INFO:tensorflow:Step 98300 per-step time 0.255s loss=0.837
    I1009 12:41:07.094725 139781364750208 model_lib_v2.py:652] Step 98300 per-step time 0.255s loss=0.837
    INFO:tensorflow:Step 98400 per-step time 0.247s loss=0.860
    I1009 12:41:31.795365 139781364750208 model_lib_v2.py:652] Step 98400 per-step time 0.247s loss=0.860
    INFO:tensorflow:Step 98500 per-step time 0.248s loss=0.809
    I1009 12:41:56.513229 139781364750208 model_lib_v2.py:652] Step 98500 per-step time 0.248s loss=0.809
    INFO:tensorflow:Step 98600 per-step time 0.242s loss=0.560
    I1009 12:42:21.390424 139781364750208 model_lib_v2.py:652] Step 98600 per-step time 0.242s loss=0.560
    INFO:tensorflow:Step 98700 per-step time 0.252s loss=0.659
    I1009 12:42:46.019719 139781364750208 model_lib_v2.py:652] Step 98700 per-step time 0.252s loss=0.659
    INFO:tensorflow:Step 98800 per-step time 0.246s loss=0.696
    I1009 12:43:10.711704 139781364750208 model_lib_v2.py:652] Step 98800 per-step time 0.246s loss=0.696
    INFO:tensorflow:Step 98900 per-step time 0.255s loss=0.976
    I1009 12:43:35.442811 139781364750208 model_lib_v2.py:652] Step 98900 per-step time 0.255s loss=0.976
    INFO:tensorflow:Step 99000 per-step time 0.241s loss=0.566
    I1009 12:44:00.017260 139781364750208 model_lib_v2.py:652] Step 99000 per-step time 0.241s loss=0.566
    INFO:tensorflow:Step 99100 per-step time 0.251s loss=0.915
    I1009 12:44:26.032156 139781364750208 model_lib_v2.py:652] Step 99100 per-step time 0.251s loss=0.915
    INFO:tensorflow:Step 99200 per-step time 0.251s loss=0.774
    I1009 12:44:50.541786 139781364750208 model_lib_v2.py:652] Step 99200 per-step time 0.251s loss=0.774
    INFO:tensorflow:Step 99300 per-step time 0.245s loss=0.945
    I1009 12:45:15.175274 139781364750208 model_lib_v2.py:652] Step 99300 per-step time 0.245s loss=0.945
    INFO:tensorflow:Step 99400 per-step time 0.240s loss=0.925
    I1009 12:45:39.907753 139781364750208 model_lib_v2.py:652] Step 99400 per-step time 0.240s loss=0.925
    INFO:tensorflow:Step 99500 per-step time 0.245s loss=0.658
    I1009 12:46:04.895078 139781364750208 model_lib_v2.py:652] Step 99500 per-step time 0.245s loss=0.658
    INFO:tensorflow:Step 99600 per-step time 0.252s loss=0.787
    I1009 12:46:29.623646 139781364750208 model_lib_v2.py:652] Step 99600 per-step time 0.252s loss=0.787
    INFO:tensorflow:Step 99700 per-step time 0.237s loss=0.872
    I1009 12:46:54.502896 139781364750208 model_lib_v2.py:652] Step 99700 per-step time 0.237s loss=0.872
    INFO:tensorflow:Step 99800 per-step time 0.250s loss=0.827
    I1009 12:47:19.194020 139781364750208 model_lib_v2.py:652] Step 99800 per-step time 0.250s loss=0.827
    INFO:tensorflow:Step 99900 per-step time 0.239s loss=0.965
    I1009 12:47:43.900100 139781364750208 model_lib_v2.py:652] Step 99900 per-step time 0.239s loss=0.965
    INFO:tensorflow:Step 100000 per-step time 0.248s loss=0.712
    I1009 12:48:08.584265 139781364750208 model_lib_v2.py:652] Step 100000 per-step time 0.248s loss=0.712
    INFO:tensorflow:Step 100100 per-step time 0.246s loss=0.782
    I1009 12:48:34.248584 139781364750208 model_lib_v2.py:652] Step 100100 per-step time 0.246s loss=0.782
    INFO:tensorflow:Step 100200 per-step time 0.253s loss=0.693
    I1009 12:48:59.058380 139781364750208 model_lib_v2.py:652] Step 100200 per-step time 0.253s loss=0.693
    INFO:tensorflow:Step 100300 per-step time 0.240s loss=0.679
    I1009 12:49:24.280009 139781364750208 model_lib_v2.py:652] Step 100300 per-step time 0.240s loss=0.679
    INFO:tensorflow:Step 100400 per-step time 0.246s loss=0.910
    I1009 12:49:49.153012 139781364750208 model_lib_v2.py:652] Step 100400 per-step time 0.246s loss=0.910
    INFO:tensorflow:Step 100500 per-step time 0.259s loss=0.476
    I1009 12:50:14.195287 139781364750208 model_lib_v2.py:652] Step 100500 per-step time 0.259s loss=0.476
    INFO:tensorflow:Step 100600 per-step time 0.254s loss=0.585
    I1009 12:50:38.924549 139781364750208 model_lib_v2.py:652] Step 100600 per-step time 0.254s loss=0.585
    INFO:tensorflow:Step 100700 per-step time 0.257s loss=0.738
    I1009 12:51:03.845216 139781364750208 model_lib_v2.py:652] Step 100700 per-step time 0.257s loss=0.738
    INFO:tensorflow:Step 100800 per-step time 0.251s loss=0.660
    I1009 12:51:28.557847 139781364750208 model_lib_v2.py:652] Step 100800 per-step time 0.251s loss=0.660
    INFO:tensorflow:Step 100900 per-step time 0.242s loss=0.401
    I1009 12:51:53.251105 139781364750208 model_lib_v2.py:652] Step 100900 per-step time 0.242s loss=0.401
    INFO:tensorflow:Step 101000 per-step time 0.249s loss=1.008
    I1009 12:52:18.041973 139781364750208 model_lib_v2.py:652] Step 101000 per-step time 0.249s loss=1.008
    INFO:tensorflow:Step 101100 per-step time 0.230s loss=0.610
    I1009 12:52:43.727766 139781364750208 model_lib_v2.py:652] Step 101100 per-step time 0.230s loss=0.610
    INFO:tensorflow:Step 101200 per-step time 0.262s loss=0.642
    I1009 12:53:08.671678 139781364750208 model_lib_v2.py:652] Step 101200 per-step time 0.262s loss=0.642
    INFO:tensorflow:Step 101300 per-step time 0.251s loss=0.696
    I1009 12:53:33.492105 139781364750208 model_lib_v2.py:652] Step 101300 per-step time 0.251s loss=0.696
    INFO:tensorflow:Step 101400 per-step time 0.233s loss=0.830
    I1009 12:53:58.396440 139781364750208 model_lib_v2.py:652] Step 101400 per-step time 0.233s loss=0.830
    INFO:tensorflow:Step 101500 per-step time 0.252s loss=0.617
    I1009 12:54:23.322256 139781364750208 model_lib_v2.py:652] Step 101500 per-step time 0.252s loss=0.617
    INFO:tensorflow:Step 101600 per-step time 0.258s loss=0.495
    I1009 12:54:48.280199 139781364750208 model_lib_v2.py:652] Step 101600 per-step time 0.258s loss=0.495
    INFO:tensorflow:Step 101700 per-step time 0.245s loss=0.778
    I1009 12:55:13.085178 139781364750208 model_lib_v2.py:652] Step 101700 per-step time 0.245s loss=0.778
    INFO:tensorflow:Step 101800 per-step time 0.248s loss=0.554
    I1009 12:55:37.894193 139781364750208 model_lib_v2.py:652] Step 101800 per-step time 0.248s loss=0.554
    INFO:tensorflow:Step 101900 per-step time 0.260s loss=0.604
    I1009 12:56:02.735726 139781364750208 model_lib_v2.py:652] Step 101900 per-step time 0.260s loss=0.604
    INFO:tensorflow:Step 102000 per-step time 0.241s loss=0.426
    I1009 12:56:27.663659 139781364750208 model_lib_v2.py:652] Step 102000 per-step time 0.241s loss=0.426
    INFO:tensorflow:Step 102100 per-step time 0.242s loss=0.885
    I1009 12:56:54.134218 139781364750208 model_lib_v2.py:652] Step 102100 per-step time 0.242s loss=0.885
    INFO:tensorflow:Step 102200 per-step time 0.239s loss=0.659
    I1009 12:57:18.829916 139781364750208 model_lib_v2.py:652] Step 102200 per-step time 0.239s loss=0.659
    INFO:tensorflow:Step 102300 per-step time 0.249s loss=0.624
    I1009 12:57:43.507858 139781364750208 model_lib_v2.py:652] Step 102300 per-step time 0.249s loss=0.624
    INFO:tensorflow:Step 102400 per-step time 0.250s loss=0.736
    I1009 12:58:07.895874 139781364750208 model_lib_v2.py:652] Step 102400 per-step time 0.250s loss=0.736
    INFO:tensorflow:Step 102500 per-step time 0.259s loss=0.919
    I1009 12:58:32.384979 139781364750208 model_lib_v2.py:652] Step 102500 per-step time 0.259s loss=0.919
    INFO:tensorflow:Step 102600 per-step time 0.236s loss=1.189
    I1009 12:58:57.071971 139781364750208 model_lib_v2.py:652] Step 102600 per-step time 0.236s loss=1.189
    INFO:tensorflow:Step 102700 per-step time 0.253s loss=0.616
    I1009 12:59:21.727160 139781364750208 model_lib_v2.py:652] Step 102700 per-step time 0.253s loss=0.616
    INFO:tensorflow:Step 102800 per-step time 0.241s loss=0.831
    I1009 12:59:46.333143 139781364750208 model_lib_v2.py:652] Step 102800 per-step time 0.241s loss=0.831
    INFO:tensorflow:Step 102900 per-step time 0.243s loss=1.122
    I1009 13:00:11.015791 139781364750208 model_lib_v2.py:652] Step 102900 per-step time 0.243s loss=1.122
    INFO:tensorflow:Step 103000 per-step time 0.238s loss=0.672
    I1009 13:00:35.646957 139781364750208 model_lib_v2.py:652] Step 103000 per-step time 0.238s loss=0.672
    INFO:tensorflow:Step 103100 per-step time 0.245s loss=0.714
    I1009 13:01:01.099928 139781364750208 model_lib_v2.py:652] Step 103100 per-step time 0.245s loss=0.714
    INFO:tensorflow:Step 103200 per-step time 0.234s loss=0.826
    I1009 13:01:25.806616 139781364750208 model_lib_v2.py:652] Step 103200 per-step time 0.234s loss=0.826
    INFO:tensorflow:Step 103300 per-step time 0.249s loss=0.683
    I1009 13:01:50.384769 139781364750208 model_lib_v2.py:652] Step 103300 per-step time 0.249s loss=0.683
    INFO:tensorflow:Step 103400 per-step time 0.237s loss=0.672
    I1009 13:02:15.092173 139781364750208 model_lib_v2.py:652] Step 103400 per-step time 0.237s loss=0.672
    INFO:tensorflow:Step 103500 per-step time 0.240s loss=0.784
    I1009 13:02:39.693648 139781364750208 model_lib_v2.py:652] Step 103500 per-step time 0.240s loss=0.784
    INFO:tensorflow:Step 103600 per-step time 0.244s loss=0.764
    I1009 13:03:04.120666 139781364750208 model_lib_v2.py:652] Step 103600 per-step time 0.244s loss=0.764
    INFO:tensorflow:Step 103700 per-step time 0.236s loss=0.872
    I1009 13:03:28.472974 139781364750208 model_lib_v2.py:652] Step 103700 per-step time 0.236s loss=0.872
    INFO:tensorflow:Step 103800 per-step time 0.242s loss=0.766
    I1009 13:03:52.974840 139781364750208 model_lib_v2.py:652] Step 103800 per-step time 0.242s loss=0.766
    INFO:tensorflow:Step 103900 per-step time 0.239s loss=0.752
    I1009 13:04:17.438045 139781364750208 model_lib_v2.py:652] Step 103900 per-step time 0.239s loss=0.752
    INFO:tensorflow:Step 104000 per-step time 0.244s loss=0.865
    I1009 13:04:41.823112 139781364750208 model_lib_v2.py:652] Step 104000 per-step time 0.244s loss=0.865
    INFO:tensorflow:Step 104100 per-step time 0.252s loss=0.686
    I1009 13:05:07.425120 139781364750208 model_lib_v2.py:652] Step 104100 per-step time 0.252s loss=0.686
    INFO:tensorflow:Step 104200 per-step time 0.243s loss=0.844
    I1009 13:05:31.863785 139781364750208 model_lib_v2.py:652] Step 104200 per-step time 0.243s loss=0.844
    INFO:tensorflow:Step 104300 per-step time 0.245s loss=0.780
    I1009 13:05:56.447132 139781364750208 model_lib_v2.py:652] Step 104300 per-step time 0.245s loss=0.780
    INFO:tensorflow:Step 104400 per-step time 0.259s loss=1.053
    I1009 13:06:21.170207 139781364750208 model_lib_v2.py:652] Step 104400 per-step time 0.259s loss=1.053
    INFO:tensorflow:Step 104500 per-step time 0.252s loss=0.917
    I1009 13:06:46.068241 139781364750208 model_lib_v2.py:652] Step 104500 per-step time 0.252s loss=0.917
    INFO:tensorflow:Step 104600 per-step time 0.233s loss=0.950
    I1009 13:07:10.613283 139781364750208 model_lib_v2.py:652] Step 104600 per-step time 0.233s loss=0.950
    INFO:tensorflow:Step 104700 per-step time 0.252s loss=0.893
    I1009 13:07:35.148712 139781364750208 model_lib_v2.py:652] Step 104700 per-step time 0.252s loss=0.893
    INFO:tensorflow:Step 104800 per-step time 0.256s loss=0.916
    I1009 13:07:59.927305 139781364750208 model_lib_v2.py:652] Step 104800 per-step time 0.256s loss=0.916
    INFO:tensorflow:Step 104900 per-step time 0.249s loss=0.416
    I1009 13:08:24.522278 139781364750208 model_lib_v2.py:652] Step 104900 per-step time 0.249s loss=0.416
    INFO:tensorflow:Step 105000 per-step time 0.245s loss=1.007
    I1009 13:08:49.338286 139781364750208 model_lib_v2.py:652] Step 105000 per-step time 0.245s loss=1.007
    INFO:tensorflow:Step 105100 per-step time 0.246s loss=0.779
    I1009 13:09:14.896185 139781364750208 model_lib_v2.py:652] Step 105100 per-step time 0.246s loss=0.779
    INFO:tensorflow:Step 105200 per-step time 0.245s loss=0.492
    I1009 13:09:39.467570 139781364750208 model_lib_v2.py:652] Step 105200 per-step time 0.245s loss=0.492
    INFO:tensorflow:Step 105300 per-step time 0.236s loss=0.777
    I1009 13:10:03.906730 139781364750208 model_lib_v2.py:652] Step 105300 per-step time 0.236s loss=0.777
    INFO:tensorflow:Step 105400 per-step time 0.249s loss=0.591
    I1009 13:10:28.385011 139781364750208 model_lib_v2.py:652] Step 105400 per-step time 0.249s loss=0.591
    INFO:tensorflow:Step 105500 per-step time 0.251s loss=0.708
    I1009 13:10:53.067174 139781364750208 model_lib_v2.py:652] Step 105500 per-step time 0.251s loss=0.708
    INFO:tensorflow:Step 105600 per-step time 0.252s loss=0.913
    I1009 13:11:17.620665 139781364750208 model_lib_v2.py:652] Step 105600 per-step time 0.252s loss=0.913
    INFO:tensorflow:Step 105700 per-step time 0.254s loss=1.125
    I1009 13:11:42.472313 139781364750208 model_lib_v2.py:652] Step 105700 per-step time 0.254s loss=1.125
    INFO:tensorflow:Step 105800 per-step time 0.246s loss=0.552
    I1009 13:12:07.354958 139781364750208 model_lib_v2.py:652] Step 105800 per-step time 0.246s loss=0.552
    INFO:tensorflow:Step 105900 per-step time 0.255s loss=0.614
    I1009 13:12:32.284146 139781364750208 model_lib_v2.py:652] Step 105900 per-step time 0.255s loss=0.614
    INFO:tensorflow:Step 106000 per-step time 0.248s loss=0.697
    I1009 13:12:57.096729 139781364750208 model_lib_v2.py:652] Step 106000 per-step time 0.248s loss=0.697
    INFO:tensorflow:Step 106100 per-step time 0.244s loss=0.579
    I1009 13:13:22.882733 139781364750208 model_lib_v2.py:652] Step 106100 per-step time 0.244s loss=0.579
    INFO:tensorflow:Step 106200 per-step time 0.241s loss=0.645
    I1009 13:13:47.590616 139781364750208 model_lib_v2.py:652] Step 106200 per-step time 0.241s loss=0.645
    INFO:tensorflow:Step 106300 per-step time 0.252s loss=0.782
    I1009 13:14:12.428799 139781364750208 model_lib_v2.py:652] Step 106300 per-step time 0.252s loss=0.782
    INFO:tensorflow:Step 106400 per-step time 0.264s loss=0.518
    I1009 13:14:37.232572 139781364750208 model_lib_v2.py:652] Step 106400 per-step time 0.264s loss=0.518
    INFO:tensorflow:Step 106500 per-step time 0.239s loss=0.822
    I1009 13:15:02.001498 139781364750208 model_lib_v2.py:652] Step 106500 per-step time 0.239s loss=0.822
    INFO:tensorflow:Step 106600 per-step time 0.242s loss=1.071
    I1009 13:15:26.727968 139781364750208 model_lib_v2.py:652] Step 106600 per-step time 0.242s loss=1.071
    INFO:tensorflow:Step 106700 per-step time 0.245s loss=0.766
    I1009 13:15:51.344561 139781364750208 model_lib_v2.py:652] Step 106700 per-step time 0.245s loss=0.766
    INFO:tensorflow:Step 106800 per-step time 0.245s loss=0.759
    I1009 13:16:15.976254 139781364750208 model_lib_v2.py:652] Step 106800 per-step time 0.245s loss=0.759
    INFO:tensorflow:Step 106900 per-step time 0.243s loss=0.976
    I1009 13:16:40.558170 139781364750208 model_lib_v2.py:652] Step 106900 per-step time 0.243s loss=0.976
    INFO:tensorflow:Step 107000 per-step time 0.232s loss=0.863
    I1009 13:17:05.624698 139781364750208 model_lib_v2.py:652] Step 107000 per-step time 0.232s loss=0.863
    INFO:tensorflow:Step 107100 per-step time 0.251s loss=0.525
    I1009 13:17:31.163267 139781364750208 model_lib_v2.py:652] Step 107100 per-step time 0.251s loss=0.525
    INFO:tensorflow:Step 107200 per-step time 0.259s loss=0.822
    I1009 13:17:55.756638 139781364750208 model_lib_v2.py:652] Step 107200 per-step time 0.259s loss=0.822
    INFO:tensorflow:Step 107300 per-step time 0.247s loss=0.603
    I1009 13:18:20.361372 139781364750208 model_lib_v2.py:652] Step 107300 per-step time 0.247s loss=0.603
    INFO:tensorflow:Step 107400 per-step time 0.250s loss=0.627
    I1009 13:18:44.877110 139781364750208 model_lib_v2.py:652] Step 107400 per-step time 0.250s loss=0.627
    INFO:tensorflow:Step 107500 per-step time 0.244s loss=0.615
    I1009 13:19:09.432793 139781364750208 model_lib_v2.py:652] Step 107500 per-step time 0.244s loss=0.615
    INFO:tensorflow:Step 107600 per-step time 0.244s loss=0.711
    I1009 13:19:34.046089 139781364750208 model_lib_v2.py:652] Step 107600 per-step time 0.244s loss=0.711
    INFO:tensorflow:Step 107700 per-step time 0.250s loss=0.648
    I1009 13:19:58.526854 139781364750208 model_lib_v2.py:652] Step 107700 per-step time 0.250s loss=0.648
    INFO:tensorflow:Step 107800 per-step time 0.245s loss=0.791
    I1009 13:20:23.002774 139781364750208 model_lib_v2.py:652] Step 107800 per-step time 0.245s loss=0.791
    INFO:tensorflow:Step 107900 per-step time 0.246s loss=0.838
    I1009 13:20:47.683676 139781364750208 model_lib_v2.py:652] Step 107900 per-step time 0.246s loss=0.838
    INFO:tensorflow:Step 108000 per-step time 0.243s loss=0.879
    I1009 13:21:12.217950 139781364750208 model_lib_v2.py:652] Step 108000 per-step time 0.243s loss=0.879
    INFO:tensorflow:Step 108100 per-step time 0.242s loss=0.526
    I1009 13:21:37.652396 139781364750208 model_lib_v2.py:652] Step 108100 per-step time 0.242s loss=0.526
    INFO:tensorflow:Step 108200 per-step time 0.256s loss=0.796
    I1009 13:22:02.400973 139781364750208 model_lib_v2.py:652] Step 108200 per-step time 0.256s loss=0.796
    INFO:tensorflow:Step 108300 per-step time 0.242s loss=1.045
    I1009 13:22:26.928455 139781364750208 model_lib_v2.py:652] Step 108300 per-step time 0.242s loss=1.045
    INFO:tensorflow:Step 108400 per-step time 0.248s loss=0.895
    I1009 13:22:51.363067 139781364750208 model_lib_v2.py:652] Step 108400 per-step time 0.248s loss=0.895
    INFO:tensorflow:Step 108500 per-step time 0.233s loss=0.544
    I1009 13:23:15.865234 139781364750208 model_lib_v2.py:652] Step 108500 per-step time 0.233s loss=0.544
    INFO:tensorflow:Step 108600 per-step time 0.238s loss=0.521
    I1009 13:23:40.285442 139781364750208 model_lib_v2.py:652] Step 108600 per-step time 0.238s loss=0.521
    INFO:tensorflow:Step 108700 per-step time 0.238s loss=0.949
    I1009 13:24:04.667580 139781364750208 model_lib_v2.py:652] Step 108700 per-step time 0.238s loss=0.949
    INFO:tensorflow:Step 108800 per-step time 0.248s loss=0.616
    I1009 13:24:28.982862 139781364750208 model_lib_v2.py:652] Step 108800 per-step time 0.248s loss=0.616
    INFO:tensorflow:Step 108900 per-step time 0.260s loss=0.895
    I1009 13:24:53.247727 139781364750208 model_lib_v2.py:652] Step 108900 per-step time 0.260s loss=0.895
    INFO:tensorflow:Step 109000 per-step time 0.231s loss=0.542
    I1009 13:25:17.600004 139781364750208 model_lib_v2.py:652] Step 109000 per-step time 0.231s loss=0.542
    INFO:tensorflow:Step 109100 per-step time 0.250s loss=0.611
    I1009 13:25:42.851703 139781364750208 model_lib_v2.py:652] Step 109100 per-step time 0.250s loss=0.611
    INFO:tensorflow:Step 109200 per-step time 0.238s loss=0.577
    I1009 13:26:07.281825 139781364750208 model_lib_v2.py:652] Step 109200 per-step time 0.238s loss=0.577
    INFO:tensorflow:Step 109300 per-step time 0.248s loss=0.779
    I1009 13:26:31.674803 139781364750208 model_lib_v2.py:652] Step 109300 per-step time 0.248s loss=0.779
    INFO:tensorflow:Step 109400 per-step time 0.250s loss=0.565
    I1009 13:26:56.153360 139781364750208 model_lib_v2.py:652] Step 109400 per-step time 0.250s loss=0.565
    INFO:tensorflow:Step 109500 per-step time 0.246s loss=0.647
    I1009 13:27:20.732594 139781364750208 model_lib_v2.py:652] Step 109500 per-step time 0.246s loss=0.647
    INFO:tensorflow:Step 109600 per-step time 0.252s loss=1.161
    I1009 13:27:45.219539 139781364750208 model_lib_v2.py:652] Step 109600 per-step time 0.252s loss=1.161
    INFO:tensorflow:Step 109700 per-step time 0.240s loss=0.899
    I1009 13:28:09.421903 139781364750208 model_lib_v2.py:652] Step 109700 per-step time 0.240s loss=0.899
    INFO:tensorflow:Step 109800 per-step time 0.252s loss=0.973
    I1009 13:28:33.706091 139781364750208 model_lib_v2.py:652] Step 109800 per-step time 0.252s loss=0.973
    INFO:tensorflow:Step 109900 per-step time 0.245s loss=0.674
    I1009 13:28:58.049043 139781364750208 model_lib_v2.py:652] Step 109900 per-step time 0.245s loss=0.674
    INFO:tensorflow:Step 110000 per-step time 0.250s loss=0.647
    I1009 13:29:22.423463 139781364750208 model_lib_v2.py:652] Step 110000 per-step time 0.250s loss=0.647
    INFO:tensorflow:Step 110100 per-step time 0.244s loss=0.753
    I1009 13:29:47.797488 139781364750208 model_lib_v2.py:652] Step 110100 per-step time 0.244s loss=0.753
    INFO:tensorflow:Step 110200 per-step time 0.253s loss=0.737
    I1009 13:30:12.195389 139781364750208 model_lib_v2.py:652] Step 110200 per-step time 0.253s loss=0.737
    INFO:tensorflow:Step 110300 per-step time 0.241s loss=0.716
    I1009 13:30:36.852573 139781364750208 model_lib_v2.py:652] Step 110300 per-step time 0.241s loss=0.716
    INFO:tensorflow:Step 110400 per-step time 0.238s loss=0.758
    I1009 13:31:01.339434 139781364750208 model_lib_v2.py:652] Step 110400 per-step time 0.238s loss=0.758
    INFO:tensorflow:Step 110500 per-step time 0.255s loss=1.075
    I1009 13:31:25.907646 139781364750208 model_lib_v2.py:652] Step 110500 per-step time 0.255s loss=1.075
    INFO:tensorflow:Step 110600 per-step time 0.252s loss=0.848
    I1009 13:31:50.595858 139781364750208 model_lib_v2.py:652] Step 110600 per-step time 0.252s loss=0.848
    INFO:tensorflow:Step 110700 per-step time 0.252s loss=0.730
    I1009 13:32:15.292088 139781364750208 model_lib_v2.py:652] Step 110700 per-step time 0.252s loss=0.730
    INFO:tensorflow:Step 110800 per-step time 0.250s loss=0.836
    I1009 13:32:40.183882 139781364750208 model_lib_v2.py:652] Step 110800 per-step time 0.250s loss=0.836
    INFO:tensorflow:Step 110900 per-step time 0.239s loss=1.031
    I1009 13:33:04.731440 139781364750208 model_lib_v2.py:652] Step 110900 per-step time 0.239s loss=1.031
    INFO:tensorflow:Step 111000 per-step time 0.245s loss=1.205
    I1009 13:33:29.308161 139781364750208 model_lib_v2.py:652] Step 111000 per-step time 0.245s loss=1.205
    INFO:tensorflow:Step 111100 per-step time 0.254s loss=0.664
    I1009 13:33:54.879194 139781364750208 model_lib_v2.py:652] Step 111100 per-step time 0.254s loss=0.664
    INFO:tensorflow:Step 111200 per-step time 0.238s loss=0.948
    I1009 13:34:19.493330 139781364750208 model_lib_v2.py:652] Step 111200 per-step time 0.238s loss=0.948
    INFO:tensorflow:Step 111300 per-step time 0.242s loss=0.607
    I1009 13:34:44.032155 139781364750208 model_lib_v2.py:652] Step 111300 per-step time 0.242s loss=0.607
    INFO:tensorflow:Step 111400 per-step time 0.234s loss=0.594
    I1009 13:35:08.438003 139781364750208 model_lib_v2.py:652] Step 111400 per-step time 0.234s loss=0.594
    INFO:tensorflow:Step 111500 per-step time 0.230s loss=0.712
    I1009 13:35:32.900096 139781364750208 model_lib_v2.py:652] Step 111500 per-step time 0.230s loss=0.712
    INFO:tensorflow:Step 111600 per-step time 0.250s loss=0.854
    I1009 13:35:57.291881 139781364750208 model_lib_v2.py:652] Step 111600 per-step time 0.250s loss=0.854
    INFO:tensorflow:Step 111700 per-step time 0.236s loss=1.048
    I1009 13:36:21.776041 139781364750208 model_lib_v2.py:652] Step 111700 per-step time 0.236s loss=1.048
    INFO:tensorflow:Step 111800 per-step time 0.256s loss=0.692
    I1009 13:36:46.189743 139781364750208 model_lib_v2.py:652] Step 111800 per-step time 0.256s loss=0.692
    INFO:tensorflow:Step 111900 per-step time 0.241s loss=0.857
    I1009 13:37:10.661198 139781364750208 model_lib_v2.py:652] Step 111900 per-step time 0.241s loss=0.857
    INFO:tensorflow:Step 112000 per-step time 0.244s loss=1.021
    I1009 13:37:35.288574 139781364750208 model_lib_v2.py:652] Step 112000 per-step time 0.244s loss=1.021
    INFO:tensorflow:Step 112100 per-step time 0.246s loss=0.709
    I1009 13:38:00.826272 139781364750208 model_lib_v2.py:652] Step 112100 per-step time 0.246s loss=0.709
    INFO:tensorflow:Step 112200 per-step time 0.234s loss=0.768
    I1009 13:38:25.229427 139781364750208 model_lib_v2.py:652] Step 112200 per-step time 0.234s loss=0.768
    INFO:tensorflow:Step 112300 per-step time 0.236s loss=0.929
    I1009 13:38:49.679078 139781364750208 model_lib_v2.py:652] Step 112300 per-step time 0.236s loss=0.929
    INFO:tensorflow:Step 112400 per-step time 0.232s loss=0.794
    I1009 13:39:14.080546 139781364750208 model_lib_v2.py:652] Step 112400 per-step time 0.232s loss=0.794
    INFO:tensorflow:Step 112500 per-step time 0.235s loss=0.692
    I1009 13:39:38.468078 139781364750208 model_lib_v2.py:652] Step 112500 per-step time 0.235s loss=0.692
    INFO:tensorflow:Step 112600 per-step time 0.250s loss=0.609
    I1009 13:40:02.999421 139781364750208 model_lib_v2.py:652] Step 112600 per-step time 0.250s loss=0.609
    INFO:tensorflow:Step 112700 per-step time 0.246s loss=0.815
    I1009 13:40:27.572081 139781364750208 model_lib_v2.py:652] Step 112700 per-step time 0.246s loss=0.815
    INFO:tensorflow:Step 112800 per-step time 0.242s loss=0.835
    I1009 13:40:51.858180 139781364750208 model_lib_v2.py:652] Step 112800 per-step time 0.242s loss=0.835
    INFO:tensorflow:Step 112900 per-step time 0.247s loss=0.851
    I1009 13:41:16.319093 139781364750208 model_lib_v2.py:652] Step 112900 per-step time 0.247s loss=0.851
    INFO:tensorflow:Step 113000 per-step time 0.249s loss=0.679
    I1009 13:41:40.966087 139781364750208 model_lib_v2.py:652] Step 113000 per-step time 0.249s loss=0.679
    INFO:tensorflow:Step 113100 per-step time 0.242s loss=0.925
    I1009 13:42:06.520062 139781364750208 model_lib_v2.py:652] Step 113100 per-step time 0.242s loss=0.925
    INFO:tensorflow:Step 113200 per-step time 0.234s loss=0.700
    I1009 13:42:31.127959 139781364750208 model_lib_v2.py:652] Step 113200 per-step time 0.234s loss=0.700
    INFO:tensorflow:Step 113300 per-step time 0.231s loss=0.669
    I1009 13:42:56.124611 139781364750208 model_lib_v2.py:652] Step 113300 per-step time 0.231s loss=0.669
    INFO:tensorflow:Step 113400 per-step time 0.231s loss=0.694
    I1009 13:43:20.826690 139781364750208 model_lib_v2.py:652] Step 113400 per-step time 0.231s loss=0.694
    INFO:tensorflow:Step 113500 per-step time 0.235s loss=0.829
    I1009 13:43:45.396065 139781364750208 model_lib_v2.py:652] Step 113500 per-step time 0.235s loss=0.829
    INFO:tensorflow:Step 113600 per-step time 0.245s loss=0.931
    I1009 13:44:10.016743 139781364750208 model_lib_v2.py:652] Step 113600 per-step time 0.245s loss=0.931
    INFO:tensorflow:Step 113700 per-step time 0.242s loss=0.623
    I1009 13:44:34.581927 139781364750208 model_lib_v2.py:652] Step 113700 per-step time 0.242s loss=0.623
    INFO:tensorflow:Step 113800 per-step time 0.245s loss=0.889
    I1009 13:44:59.298194 139781364750208 model_lib_v2.py:652] Step 113800 per-step time 0.245s loss=0.889
    INFO:tensorflow:Step 113900 per-step time 0.248s loss=0.777
    I1009 13:45:23.806219 139781364750208 model_lib_v2.py:652] Step 113900 per-step time 0.248s loss=0.777
    INFO:tensorflow:Step 114000 per-step time 0.253s loss=0.871
    I1009 13:45:48.430248 139781364750208 model_lib_v2.py:652] Step 114000 per-step time 0.253s loss=0.871
    INFO:tensorflow:Step 114100 per-step time 0.255s loss=0.839
    I1009 13:46:14.692820 139781364750208 model_lib_v2.py:652] Step 114100 per-step time 0.255s loss=0.839
    INFO:tensorflow:Step 114200 per-step time 0.246s loss=0.814
    I1009 13:46:39.156429 139781364750208 model_lib_v2.py:652] Step 114200 per-step time 0.246s loss=0.814
    INFO:tensorflow:Step 114300 per-step time 0.248s loss=1.336
    I1009 13:47:03.770791 139781364750208 model_lib_v2.py:652] Step 114300 per-step time 0.248s loss=1.336
    INFO:tensorflow:Step 114400 per-step time 0.248s loss=0.602
    I1009 13:47:28.154982 139781364750208 model_lib_v2.py:652] Step 114400 per-step time 0.248s loss=0.602
    INFO:tensorflow:Step 114500 per-step time 0.250s loss=0.724
    I1009 13:47:53.037302 139781364750208 model_lib_v2.py:652] Step 114500 per-step time 0.250s loss=0.724
    INFO:tensorflow:Step 114600 per-step time 0.246s loss=0.941
    I1009 13:48:17.897771 139781364750208 model_lib_v2.py:652] Step 114600 per-step time 0.246s loss=0.941
    INFO:tensorflow:Step 114700 per-step time 0.245s loss=0.813
    I1009 13:48:42.646659 139781364750208 model_lib_v2.py:652] Step 114700 per-step time 0.245s loss=0.813
    INFO:tensorflow:Step 114800 per-step time 0.246s loss=0.599
    I1009 13:49:07.311248 139781364750208 model_lib_v2.py:652] Step 114800 per-step time 0.246s loss=0.599
    INFO:tensorflow:Step 114900 per-step time 0.239s loss=0.660
    I1009 13:49:31.711712 139781364750208 model_lib_v2.py:652] Step 114900 per-step time 0.239s loss=0.660
    INFO:tensorflow:Step 115000 per-step time 0.244s loss=0.881
    I1009 13:49:56.163605 139781364750208 model_lib_v2.py:652] Step 115000 per-step time 0.244s loss=0.881
    INFO:tensorflow:Step 115100 per-step time 0.252s loss=0.853
    I1009 13:50:21.647248 139781364750208 model_lib_v2.py:652] Step 115100 per-step time 0.252s loss=0.853
    INFO:tensorflow:Step 115200 per-step time 0.241s loss=0.684
    I1009 13:50:46.280369 139781364750208 model_lib_v2.py:652] Step 115200 per-step time 0.241s loss=0.684
    INFO:tensorflow:Step 115300 per-step time 0.239s loss=0.383
    I1009 13:51:10.875227 139781364750208 model_lib_v2.py:652] Step 115300 per-step time 0.239s loss=0.383
    INFO:tensorflow:Step 115400 per-step time 0.241s loss=1.073
    I1009 13:51:35.242840 139781364750208 model_lib_v2.py:652] Step 115400 per-step time 0.241s loss=1.073
    INFO:tensorflow:Step 115500 per-step time 0.245s loss=0.720
    I1009 13:51:59.857325 139781364750208 model_lib_v2.py:652] Step 115500 per-step time 0.245s loss=0.720
    INFO:tensorflow:Step 115600 per-step time 0.253s loss=0.727
    I1009 13:52:24.407000 139781364750208 model_lib_v2.py:652] Step 115600 per-step time 0.253s loss=0.727
    INFO:tensorflow:Step 115700 per-step time 0.250s loss=0.437
    I1009 13:52:48.897889 139781364750208 model_lib_v2.py:652] Step 115700 per-step time 0.250s loss=0.437
    INFO:tensorflow:Step 115800 per-step time 0.246s loss=0.853
    I1009 13:53:13.670720 139781364750208 model_lib_v2.py:652] Step 115800 per-step time 0.246s loss=0.853
    INFO:tensorflow:Step 115900 per-step time 0.241s loss=0.759
    I1009 13:53:38.235083 139781364750208 model_lib_v2.py:652] Step 115900 per-step time 0.241s loss=0.759
    INFO:tensorflow:Step 116000 per-step time 0.259s loss=0.516
    I1009 13:54:02.675775 139781364750208 model_lib_v2.py:652] Step 116000 per-step time 0.259s loss=0.516
    INFO:tensorflow:Step 116100 per-step time 0.243s loss=0.545
    I1009 13:54:28.158579 139781364750208 model_lib_v2.py:652] Step 116100 per-step time 0.243s loss=0.545
    INFO:tensorflow:Step 116200 per-step time 0.242s loss=0.787
    I1009 13:54:52.650090 139781364750208 model_lib_v2.py:652] Step 116200 per-step time 0.242s loss=0.787
    INFO:tensorflow:Step 116300 per-step time 0.230s loss=0.629
    I1009 13:55:16.921385 139781364750208 model_lib_v2.py:652] Step 116300 per-step time 0.230s loss=0.629
    INFO:tensorflow:Step 116400 per-step time 0.246s loss=0.789
    I1009 13:55:41.324811 139781364750208 model_lib_v2.py:652] Step 116400 per-step time 0.246s loss=0.789
    INFO:tensorflow:Step 116500 per-step time 0.245s loss=0.879
    I1009 13:56:05.903378 139781364750208 model_lib_v2.py:652] Step 116500 per-step time 0.245s loss=0.879
    INFO:tensorflow:Step 116600 per-step time 0.242s loss=0.869
    I1009 13:56:30.503605 139781364750208 model_lib_v2.py:652] Step 116600 per-step time 0.242s loss=0.869
    INFO:tensorflow:Step 116700 per-step time 0.247s loss=0.690
    I1009 13:56:54.926112 139781364750208 model_lib_v2.py:652] Step 116700 per-step time 0.247s loss=0.690
    INFO:tensorflow:Step 116800 per-step time 0.247s loss=0.798
    I1009 13:57:19.408313 139781364750208 model_lib_v2.py:652] Step 116800 per-step time 0.247s loss=0.798
    INFO:tensorflow:Step 116900 per-step time 0.241s loss=0.827
    I1009 13:57:43.932592 139781364750208 model_lib_v2.py:652] Step 116900 per-step time 0.241s loss=0.827
    INFO:tensorflow:Step 117000 per-step time 0.252s loss=0.762
    I1009 13:58:08.358468 139781364750208 model_lib_v2.py:652] Step 117000 per-step time 0.252s loss=0.762
    INFO:tensorflow:Step 117100 per-step time 0.247s loss=0.685
    I1009 13:58:33.826262 139781364750208 model_lib_v2.py:652] Step 117100 per-step time 0.247s loss=0.685
    INFO:tensorflow:Step 117200 per-step time 0.237s loss=1.014
    I1009 13:58:58.062068 139781364750208 model_lib_v2.py:652] Step 117200 per-step time 0.237s loss=1.014
    INFO:tensorflow:Step 117300 per-step time 0.239s loss=0.711
    I1009 13:59:22.478806 139781364750208 model_lib_v2.py:652] Step 117300 per-step time 0.239s loss=0.711
    INFO:tensorflow:Step 117400 per-step time 0.254s loss=0.857
    I1009 13:59:46.951263 139781364750208 model_lib_v2.py:652] Step 117400 per-step time 0.254s loss=0.857
    INFO:tensorflow:Step 117500 per-step time 0.248s loss=0.707
    I1009 14:00:11.347252 139781364750208 model_lib_v2.py:652] Step 117500 per-step time 0.248s loss=0.707
    INFO:tensorflow:Step 117600 per-step time 0.237s loss=0.645
    I1009 14:00:35.533854 139781364750208 model_lib_v2.py:652] Step 117600 per-step time 0.237s loss=0.645
    INFO:tensorflow:Step 117700 per-step time 0.248s loss=0.818
    I1009 14:00:59.774037 139781364750208 model_lib_v2.py:652] Step 117700 per-step time 0.248s loss=0.818
    INFO:tensorflow:Step 117800 per-step time 0.246s loss=1.049
    I1009 14:01:23.940067 139781364750208 model_lib_v2.py:652] Step 117800 per-step time 0.246s loss=1.049
    INFO:tensorflow:Step 117900 per-step time 0.244s loss=1.047
    I1009 14:01:48.205002 139781364750208 model_lib_v2.py:652] Step 117900 per-step time 0.244s loss=1.047
    INFO:tensorflow:Step 118000 per-step time 0.242s loss=0.465
    I1009 14:02:12.367722 139781364750208 model_lib_v2.py:652] Step 118000 per-step time 0.242s loss=0.465
    INFO:tensorflow:Step 118100 per-step time 0.238s loss=0.791
    I1009 14:02:37.675586 139781364750208 model_lib_v2.py:652] Step 118100 per-step time 0.238s loss=0.791
    INFO:tensorflow:Step 118200 per-step time 0.256s loss=0.717
    I1009 14:03:02.068804 139781364750208 model_lib_v2.py:652] Step 118200 per-step time 0.256s loss=0.717
    INFO:tensorflow:Step 118300 per-step time 0.252s loss=0.721
    I1009 14:03:26.371896 139781364750208 model_lib_v2.py:652] Step 118300 per-step time 0.252s loss=0.721
    INFO:tensorflow:Step 118400 per-step time 0.252s loss=0.648
    I1009 14:03:50.820877 139781364750208 model_lib_v2.py:652] Step 118400 per-step time 0.252s loss=0.648
    INFO:tensorflow:Step 118500 per-step time 0.232s loss=0.851
    I1009 14:04:15.255380 139781364750208 model_lib_v2.py:652] Step 118500 per-step time 0.232s loss=0.851
    INFO:tensorflow:Step 118600 per-step time 0.256s loss=0.865
    I1009 14:04:39.733285 139781364750208 model_lib_v2.py:652] Step 118600 per-step time 0.256s loss=0.865
    INFO:tensorflow:Step 118700 per-step time 0.237s loss=0.902
    I1009 14:05:04.175707 139781364750208 model_lib_v2.py:652] Step 118700 per-step time 0.237s loss=0.902
    INFO:tensorflow:Step 118800 per-step time 0.246s loss=0.863
    I1009 14:05:28.586560 139781364750208 model_lib_v2.py:652] Step 118800 per-step time 0.246s loss=0.863
    INFO:tensorflow:Step 118900 per-step time 0.253s loss=0.895
    I1009 14:05:52.910004 139781364750208 model_lib_v2.py:652] Step 118900 per-step time 0.253s loss=0.895
    INFO:tensorflow:Step 119000 per-step time 0.237s loss=0.797
    I1009 14:06:17.230005 139781364750208 model_lib_v2.py:652] Step 119000 per-step time 0.237s loss=0.797
    INFO:tensorflow:Step 119100 per-step time 0.243s loss=0.825
    I1009 14:06:42.305095 139781364750208 model_lib_v2.py:652] Step 119100 per-step time 0.243s loss=0.825
    INFO:tensorflow:Step 119200 per-step time 0.245s loss=0.949
    I1009 14:07:06.588851 139781364750208 model_lib_v2.py:652] Step 119200 per-step time 0.245s loss=0.949
    INFO:tensorflow:Step 119300 per-step time 0.240s loss=0.826
    I1009 14:07:30.927655 139781364750208 model_lib_v2.py:652] Step 119300 per-step time 0.240s loss=0.826
    INFO:tensorflow:Step 119400 per-step time 0.240s loss=0.805
    I1009 14:07:55.160233 139781364750208 model_lib_v2.py:652] Step 119400 per-step time 0.240s loss=0.805
    INFO:tensorflow:Step 119500 per-step time 0.240s loss=0.835
    I1009 14:08:19.488151 139781364750208 model_lib_v2.py:652] Step 119500 per-step time 0.240s loss=0.835
    INFO:tensorflow:Step 119600 per-step time 0.252s loss=1.038
    I1009 14:08:43.934009 139781364750208 model_lib_v2.py:652] Step 119600 per-step time 0.252s loss=1.038
    INFO:tensorflow:Step 119700 per-step time 0.258s loss=0.585
    I1009 14:09:08.305160 139781364750208 model_lib_v2.py:652] Step 119700 per-step time 0.258s loss=0.585
    INFO:tensorflow:Step 119800 per-step time 0.243s loss=0.363
    I1009 14:09:33.032850 139781364750208 model_lib_v2.py:652] Step 119800 per-step time 0.243s loss=0.363
    INFO:tensorflow:Step 119900 per-step time 0.254s loss=0.985
    I1009 14:09:57.581310 139781364750208 model_lib_v2.py:652] Step 119900 per-step time 0.254s loss=0.985
    INFO:tensorflow:Step 120000 per-step time 0.244s loss=0.641
    I1009 14:10:22.425264 139781364750208 model_lib_v2.py:652] Step 120000 per-step time 0.244s loss=0.641
    INFO:tensorflow:Step 120100 per-step time 0.243s loss=0.992
    I1009 14:10:48.134929 139781364750208 model_lib_v2.py:652] Step 120100 per-step time 0.243s loss=0.992
    INFO:tensorflow:Step 120200 per-step time 0.256s loss=0.888
    I1009 14:11:12.703199 139781364750208 model_lib_v2.py:652] Step 120200 per-step time 0.256s loss=0.888
    INFO:tensorflow:Step 120300 per-step time 0.241s loss=1.112
    I1009 14:11:37.364937 139781364750208 model_lib_v2.py:652] Step 120300 per-step time 0.241s loss=1.112
    INFO:tensorflow:Step 120400 per-step time 0.245s loss=0.782
    I1009 14:12:01.896634 139781364750208 model_lib_v2.py:652] Step 120400 per-step time 0.245s loss=0.782
    INFO:tensorflow:Step 120500 per-step time 0.255s loss=0.811
    I1009 14:12:26.499932 139781364750208 model_lib_v2.py:652] Step 120500 per-step time 0.255s loss=0.811
    INFO:tensorflow:Step 120600 per-step time 0.241s loss=0.548
    I1009 14:12:51.050578 139781364750208 model_lib_v2.py:652] Step 120600 per-step time 0.241s loss=0.548
    INFO:tensorflow:Step 120700 per-step time 0.249s loss=0.940
    I1009 14:13:15.656743 139781364750208 model_lib_v2.py:652] Step 120700 per-step time 0.249s loss=0.940
    INFO:tensorflow:Step 120800 per-step time 0.248s loss=0.981
    I1009 14:13:40.061769 139781364750208 model_lib_v2.py:652] Step 120800 per-step time 0.248s loss=0.981
    INFO:tensorflow:Step 120900 per-step time 0.245s loss=0.662
    I1009 14:14:05.011070 139781364750208 model_lib_v2.py:652] Step 120900 per-step time 0.245s loss=0.662
    INFO:tensorflow:Step 121000 per-step time 0.248s loss=0.772
    I1009 14:14:29.639903 139781364750208 model_lib_v2.py:652] Step 121000 per-step time 0.248s loss=0.772
    INFO:tensorflow:Step 121100 per-step time 0.240s loss=1.193
    I1009 14:14:55.014108 139781364750208 model_lib_v2.py:652] Step 121100 per-step time 0.240s loss=1.193
    INFO:tensorflow:Step 121200 per-step time 0.247s loss=0.671
    I1009 14:15:19.467975 139781364750208 model_lib_v2.py:652] Step 121200 per-step time 0.247s loss=0.671
    INFO:tensorflow:Step 121300 per-step time 0.241s loss=0.780
    I1009 14:15:43.973946 139781364750208 model_lib_v2.py:652] Step 121300 per-step time 0.241s loss=0.780
    INFO:tensorflow:Step 121400 per-step time 0.238s loss=0.594
    I1009 14:16:08.502681 139781364750208 model_lib_v2.py:652] Step 121400 per-step time 0.238s loss=0.594
    INFO:tensorflow:Step 121500 per-step time 0.247s loss=0.607
    I1009 14:16:33.193138 139781364750208 model_lib_v2.py:652] Step 121500 per-step time 0.247s loss=0.607
    INFO:tensorflow:Step 121600 per-step time 0.248s loss=1.021
    I1009 14:16:57.757927 139781364750208 model_lib_v2.py:652] Step 121600 per-step time 0.248s loss=1.021
    INFO:tensorflow:Step 121700 per-step time 0.243s loss=0.787
    I1009 14:17:22.087854 139781364750208 model_lib_v2.py:652] Step 121700 per-step time 0.243s loss=0.787
    INFO:tensorflow:Step 121800 per-step time 0.245s loss=0.949
    I1009 14:17:46.569636 139781364750208 model_lib_v2.py:652] Step 121800 per-step time 0.245s loss=0.949
    INFO:tensorflow:Step 121900 per-step time 0.240s loss=0.731
    I1009 14:18:10.920642 139781364750208 model_lib_v2.py:652] Step 121900 per-step time 0.240s loss=0.731
    INFO:tensorflow:Step 122000 per-step time 0.257s loss=0.516
    I1009 14:18:35.186696 139781364750208 model_lib_v2.py:652] Step 122000 per-step time 0.257s loss=0.516
    INFO:tensorflow:Step 122100 per-step time 0.242s loss=0.655
    I1009 14:19:00.594455 139781364750208 model_lib_v2.py:652] Step 122100 per-step time 0.242s loss=0.655
    INFO:tensorflow:Step 122200 per-step time 0.256s loss=0.672
    I1009 14:19:25.144335 139781364750208 model_lib_v2.py:652] Step 122200 per-step time 0.256s loss=0.672
    INFO:tensorflow:Step 122300 per-step time 0.237s loss=0.618
    I1009 14:19:49.598328 139781364750208 model_lib_v2.py:652] Step 122300 per-step time 0.237s loss=0.618
    INFO:tensorflow:Step 122400 per-step time 0.238s loss=0.736
    I1009 14:20:13.924951 139781364750208 model_lib_v2.py:652] Step 122400 per-step time 0.238s loss=0.736
    INFO:tensorflow:Step 122500 per-step time 0.247s loss=0.870
    I1009 14:20:38.067144 139781364750208 model_lib_v2.py:652] Step 122500 per-step time 0.247s loss=0.870
    INFO:tensorflow:Step 122600 per-step time 0.255s loss=0.778
    I1009 14:21:02.436158 139781364750208 model_lib_v2.py:652] Step 122600 per-step time 0.255s loss=0.778
    INFO:tensorflow:Step 122700 per-step time 0.250s loss=0.691
    I1009 14:21:26.665274 139781364750208 model_lib_v2.py:652] Step 122700 per-step time 0.250s loss=0.691
    INFO:tensorflow:Step 122800 per-step time 0.231s loss=0.590
    I1009 14:21:50.949069 139781364750208 model_lib_v2.py:652] Step 122800 per-step time 0.231s loss=0.590
    INFO:tensorflow:Step 122900 per-step time 0.241s loss=0.682
    I1009 14:22:15.235562 139781364750208 model_lib_v2.py:652] Step 122900 per-step time 0.241s loss=0.682
    INFO:tensorflow:Step 123000 per-step time 0.238s loss=0.448
    I1009 14:22:39.438767 139781364750208 model_lib_v2.py:652] Step 123000 per-step time 0.238s loss=0.448
    INFO:tensorflow:Step 123100 per-step time 0.234s loss=1.171
    I1009 14:23:04.685386 139781364750208 model_lib_v2.py:652] Step 123100 per-step time 0.234s loss=1.171
    INFO:tensorflow:Step 123200 per-step time 0.245s loss=0.919
    I1009 14:23:29.108980 139781364750208 model_lib_v2.py:652] Step 123200 per-step time 0.245s loss=0.919
    INFO:tensorflow:Step 123300 per-step time 0.234s loss=0.960
    I1009 14:23:53.214967 139781364750208 model_lib_v2.py:652] Step 123300 per-step time 0.234s loss=0.960
    INFO:tensorflow:Step 123400 per-step time 0.253s loss=0.674
    I1009 14:24:17.558127 139781364750208 model_lib_v2.py:652] Step 123400 per-step time 0.253s loss=0.674
    INFO:tensorflow:Step 123500 per-step time 0.249s loss=0.582
    I1009 14:24:41.634478 139781364750208 model_lib_v2.py:652] Step 123500 per-step time 0.249s loss=0.582
    INFO:tensorflow:Step 123600 per-step time 0.231s loss=0.901
    I1009 14:25:05.679799 139781364750208 model_lib_v2.py:652] Step 123600 per-step time 0.231s loss=0.901
    INFO:tensorflow:Step 123700 per-step time 0.237s loss=0.860
    I1009 14:25:29.764953 139781364750208 model_lib_v2.py:652] Step 123700 per-step time 0.237s loss=0.860
    INFO:tensorflow:Step 123800 per-step time 0.248s loss=0.631
    I1009 14:25:54.032283 139781364750208 model_lib_v2.py:652] Step 123800 per-step time 0.248s loss=0.631
    INFO:tensorflow:Step 123900 per-step time 0.240s loss=0.807
    I1009 14:26:18.397279 139781364750208 model_lib_v2.py:652] Step 123900 per-step time 0.240s loss=0.807
    INFO:tensorflow:Step 124000 per-step time 0.246s loss=0.767
    I1009 14:26:42.584523 139781364750208 model_lib_v2.py:652] Step 124000 per-step time 0.246s loss=0.767
    INFO:tensorflow:Step 124100 per-step time 0.243s loss=0.545
    I1009 14:27:07.960337 139781364750208 model_lib_v2.py:652] Step 124100 per-step time 0.243s loss=0.545
    INFO:tensorflow:Step 124200 per-step time 0.238s loss=0.694
    I1009 14:27:32.362792 139781364750208 model_lib_v2.py:652] Step 124200 per-step time 0.238s loss=0.694
    INFO:tensorflow:Step 124300 per-step time 0.245s loss=0.768
    I1009 14:27:56.703423 139781364750208 model_lib_v2.py:652] Step 124300 per-step time 0.245s loss=0.768
    INFO:tensorflow:Step 124400 per-step time 0.242s loss=0.770
    I1009 14:28:21.162953 139781364750208 model_lib_v2.py:652] Step 124400 per-step time 0.242s loss=0.770
    INFO:tensorflow:Step 124500 per-step time 0.248s loss=1.102
    I1009 14:28:45.605989 139781364750208 model_lib_v2.py:652] Step 124500 per-step time 0.248s loss=1.102
    INFO:tensorflow:Step 124600 per-step time 0.240s loss=0.941
    I1009 14:29:10.169039 139781364750208 model_lib_v2.py:652] Step 124600 per-step time 0.240s loss=0.941
    INFO:tensorflow:Step 124700 per-step time 0.250s loss=1.122
    I1009 14:29:34.870569 139781364750208 model_lib_v2.py:652] Step 124700 per-step time 0.250s loss=1.122
    INFO:tensorflow:Step 124800 per-step time 0.243s loss=0.761
    I1009 14:29:59.205204 139781364750208 model_lib_v2.py:652] Step 124800 per-step time 0.243s loss=0.761
    INFO:tensorflow:Step 124900 per-step time 0.242s loss=0.739
    I1009 14:30:23.787704 139781364750208 model_lib_v2.py:652] Step 124900 per-step time 0.242s loss=0.739
    INFO:tensorflow:Step 125000 per-step time 0.247s loss=0.827
    I1009 14:30:48.283972 139781364750208 model_lib_v2.py:652] Step 125000 per-step time 0.247s loss=0.827
    INFO:tensorflow:Step 125100 per-step time 0.250s loss=0.807
    I1009 14:31:13.605146 139781364750208 model_lib_v2.py:652] Step 125100 per-step time 0.250s loss=0.807
    INFO:tensorflow:Step 125200 per-step time 0.242s loss=0.589
    I1009 14:31:37.770448 139781364750208 model_lib_v2.py:652] Step 125200 per-step time 0.242s loss=0.589
    INFO:tensorflow:Step 125300 per-step time 0.253s loss=0.723
    I1009 14:32:02.110473 139781364750208 model_lib_v2.py:652] Step 125300 per-step time 0.253s loss=0.723
    INFO:tensorflow:Step 125400 per-step time 0.243s loss=0.723
    I1009 14:32:26.465925 139781364750208 model_lib_v2.py:652] Step 125400 per-step time 0.243s loss=0.723
    INFO:tensorflow:Step 125500 per-step time 0.241s loss=0.871
    I1009 14:32:50.869091 139781364750208 model_lib_v2.py:652] Step 125500 per-step time 0.241s loss=0.871
    INFO:tensorflow:Step 125600 per-step time 0.230s loss=0.970
    I1009 14:33:15.190738 139781364750208 model_lib_v2.py:652] Step 125600 per-step time 0.230s loss=0.970
    INFO:tensorflow:Step 125700 per-step time 0.242s loss=0.672
    I1009 14:33:39.569100 139781364750208 model_lib_v2.py:652] Step 125700 per-step time 0.242s loss=0.672
    INFO:tensorflow:Step 125800 per-step time 0.250s loss=0.628
    I1009 14:34:03.968024 139781364750208 model_lib_v2.py:652] Step 125800 per-step time 0.250s loss=0.628
    INFO:tensorflow:Step 125900 per-step time 0.243s loss=0.635
    I1009 14:34:28.324402 139781364750208 model_lib_v2.py:652] Step 125900 per-step time 0.243s loss=0.635
    INFO:tensorflow:Step 126000 per-step time 0.237s loss=0.767
    I1009 14:34:52.900885 139781364750208 model_lib_v2.py:652] Step 126000 per-step time 0.237s loss=0.767
    INFO:tensorflow:Step 126100 per-step time 0.251s loss=0.950
    I1009 14:35:18.312702 139781364750208 model_lib_v2.py:652] Step 126100 per-step time 0.251s loss=0.950
    INFO:tensorflow:Step 126200 per-step time 0.241s loss=0.832
    I1009 14:35:42.612993 139781364750208 model_lib_v2.py:652] Step 126200 per-step time 0.241s loss=0.832
    INFO:tensorflow:Step 126300 per-step time 0.243s loss=0.717
    I1009 14:36:07.051240 139781364750208 model_lib_v2.py:652] Step 126300 per-step time 0.243s loss=0.717
    INFO:tensorflow:Step 126400 per-step time 0.241s loss=0.720
    I1009 14:36:31.552677 139781364750208 model_lib_v2.py:652] Step 126400 per-step time 0.241s loss=0.720
    INFO:tensorflow:Step 126500 per-step time 0.242s loss=0.737
    I1009 14:36:56.097746 139781364750208 model_lib_v2.py:652] Step 126500 per-step time 0.242s loss=0.737
    INFO:tensorflow:Step 126600 per-step time 0.234s loss=0.645
    I1009 14:37:20.620939 139781364750208 model_lib_v2.py:652] Step 126600 per-step time 0.234s loss=0.645
    INFO:tensorflow:Step 126700 per-step time 0.251s loss=0.689
    I1009 14:37:45.145399 139781364750208 model_lib_v2.py:652] Step 126700 per-step time 0.251s loss=0.689
    INFO:tensorflow:Step 126800 per-step time 0.259s loss=0.590
    I1009 14:38:09.526480 139781364750208 model_lib_v2.py:652] Step 126800 per-step time 0.259s loss=0.590
    INFO:tensorflow:Step 126900 per-step time 0.256s loss=0.952
    I1009 14:38:33.888209 139781364750208 model_lib_v2.py:652] Step 126900 per-step time 0.256s loss=0.952
    INFO:tensorflow:Step 127000 per-step time 0.238s loss=0.737
    I1009 14:38:58.271206 139781364750208 model_lib_v2.py:652] Step 127000 per-step time 0.238s loss=0.737
    INFO:tensorflow:Step 127100 per-step time 0.243s loss=0.514
    I1009 14:39:23.762522 139781364750208 model_lib_v2.py:652] Step 127100 per-step time 0.243s loss=0.514
    INFO:tensorflow:Step 127200 per-step time 0.254s loss=0.691
    I1009 14:39:48.326144 139781364750208 model_lib_v2.py:652] Step 127200 per-step time 0.254s loss=0.691
    INFO:tensorflow:Step 127300 per-step time 0.251s loss=0.806
    I1009 14:40:12.590442 139781364750208 model_lib_v2.py:652] Step 127300 per-step time 0.251s loss=0.806
    INFO:tensorflow:Step 127400 per-step time 0.239s loss=0.833
    I1009 14:40:36.880059 139781364750208 model_lib_v2.py:652] Step 127400 per-step time 0.239s loss=0.833
    INFO:tensorflow:Step 127500 per-step time 0.240s loss=0.703
    I1009 14:41:01.001068 139781364750208 model_lib_v2.py:652] Step 127500 per-step time 0.240s loss=0.703
    INFO:tensorflow:Step 127600 per-step time 0.252s loss=0.497
    I1009 14:41:25.095334 139781364750208 model_lib_v2.py:652] Step 127600 per-step time 0.252s loss=0.497
    INFO:tensorflow:Step 127700 per-step time 0.243s loss=0.447
    I1009 14:41:49.404149 139781364750208 model_lib_v2.py:652] Step 127700 per-step time 0.243s loss=0.447
    INFO:tensorflow:Step 127800 per-step time 0.232s loss=0.654
    I1009 14:42:13.548309 139781364750208 model_lib_v2.py:652] Step 127800 per-step time 0.232s loss=0.654
    INFO:tensorflow:Step 127900 per-step time 0.241s loss=0.626
    I1009 14:42:37.862708 139781364750208 model_lib_v2.py:652] Step 127900 per-step time 0.241s loss=0.626
    INFO:tensorflow:Step 128000 per-step time 0.249s loss=0.785
    I1009 14:43:02.223341 139781364750208 model_lib_v2.py:652] Step 128000 per-step time 0.249s loss=0.785
    INFO:tensorflow:Step 128100 per-step time 0.247s loss=0.661
    I1009 14:43:27.275259 139781364750208 model_lib_v2.py:652] Step 128100 per-step time 0.247s loss=0.661
    INFO:tensorflow:Step 128200 per-step time 0.236s loss=0.585
    I1009 14:43:51.585547 139781364750208 model_lib_v2.py:652] Step 128200 per-step time 0.236s loss=0.585
    INFO:tensorflow:Step 128300 per-step time 0.236s loss=0.691
    I1009 14:44:15.740133 139781364750208 model_lib_v2.py:652] Step 128300 per-step time 0.236s loss=0.691
    INFO:tensorflow:Step 128400 per-step time 0.238s loss=0.663
    I1009 14:44:39.936904 139781364750208 model_lib_v2.py:652] Step 128400 per-step time 0.238s loss=0.663
    INFO:tensorflow:Step 128500 per-step time 0.246s loss=0.684
    I1009 14:45:04.506760 139781364750208 model_lib_v2.py:652] Step 128500 per-step time 0.246s loss=0.684
    INFO:tensorflow:Step 128600 per-step time 0.247s loss=0.770
    I1009 14:45:28.850183 139781364750208 model_lib_v2.py:652] Step 128600 per-step time 0.247s loss=0.770
    INFO:tensorflow:Step 128700 per-step time 0.243s loss=0.848
    I1009 14:45:53.241798 139781364750208 model_lib_v2.py:652] Step 128700 per-step time 0.243s loss=0.848
    INFO:tensorflow:Step 128800 per-step time 0.235s loss=0.651
    I1009 14:46:17.552838 139781364750208 model_lib_v2.py:652] Step 128800 per-step time 0.235s loss=0.651
    INFO:tensorflow:Step 128900 per-step time 0.246s loss=0.908
    I1009 14:46:41.663426 139781364750208 model_lib_v2.py:652] Step 128900 per-step time 0.246s loss=0.908
    INFO:tensorflow:Step 129000 per-step time 0.253s loss=0.690
    I1009 14:47:05.927577 139781364750208 model_lib_v2.py:652] Step 129000 per-step time 0.253s loss=0.690
    INFO:tensorflow:Step 129100 per-step time 0.238s loss=0.537
    I1009 14:47:30.990546 139781364750208 model_lib_v2.py:652] Step 129100 per-step time 0.238s loss=0.537
    INFO:tensorflow:Step 129200 per-step time 0.247s loss=0.717
    I1009 14:47:55.216536 139781364750208 model_lib_v2.py:652] Step 129200 per-step time 0.247s loss=0.717
    INFO:tensorflow:Step 129300 per-step time 0.240s loss=0.644
    I1009 14:48:19.342116 139781364750208 model_lib_v2.py:652] Step 129300 per-step time 0.240s loss=0.644
    INFO:tensorflow:Step 129400 per-step time 0.246s loss=0.663
    I1009 14:48:43.708752 139781364750208 model_lib_v2.py:652] Step 129400 per-step time 0.246s loss=0.663
    INFO:tensorflow:Step 129500 per-step time 0.245s loss=0.479
    I1009 14:49:08.027857 139781364750208 model_lib_v2.py:652] Step 129500 per-step time 0.245s loss=0.479
    INFO:tensorflow:Step 129600 per-step time 0.236s loss=0.547
    I1009 14:49:32.186786 139781364750208 model_lib_v2.py:652] Step 129600 per-step time 0.236s loss=0.547
    INFO:tensorflow:Step 129700 per-step time 0.244s loss=0.822
    I1009 14:49:56.417839 139781364750208 model_lib_v2.py:652] Step 129700 per-step time 0.244s loss=0.822
    INFO:tensorflow:Step 129800 per-step time 0.256s loss=0.558
    I1009 14:50:21.014090 139781364750208 model_lib_v2.py:652] Step 129800 per-step time 0.256s loss=0.558
    INFO:tensorflow:Step 129900 per-step time 0.243s loss=0.734
    I1009 14:50:45.038247 139781364750208 model_lib_v2.py:652] Step 129900 per-step time 0.243s loss=0.734
    INFO:tensorflow:Step 130000 per-step time 0.249s loss=0.587
    I1009 14:51:09.260608 139781364750208 model_lib_v2.py:652] Step 130000 per-step time 0.249s loss=0.587
    INFO:tensorflow:Step 130100 per-step time 0.240s loss=0.807
    I1009 14:51:34.236303 139781364750208 model_lib_v2.py:652] Step 130100 per-step time 0.240s loss=0.807
    INFO:tensorflow:Step 130200 per-step time 0.246s loss=1.049
    I1009 14:51:58.365578 139781364750208 model_lib_v2.py:652] Step 130200 per-step time 0.246s loss=1.049
    INFO:tensorflow:Step 130300 per-step time 0.250s loss=0.603
    I1009 14:52:22.557245 139781364750208 model_lib_v2.py:652] Step 130300 per-step time 0.250s loss=0.603
    INFO:tensorflow:Step 130400 per-step time 0.246s loss=0.825
    I1009 14:52:47.209479 139781364750208 model_lib_v2.py:652] Step 130400 per-step time 0.246s loss=0.825
    INFO:tensorflow:Step 130500 per-step time 0.250s loss=0.624
    I1009 14:53:11.818126 139781364750208 model_lib_v2.py:652] Step 130500 per-step time 0.250s loss=0.624
    INFO:tensorflow:Step 130600 per-step time 0.232s loss=0.725
    I1009 14:53:36.378334 139781364750208 model_lib_v2.py:652] Step 130600 per-step time 0.232s loss=0.725
    INFO:tensorflow:Step 130700 per-step time 0.254s loss=0.715
    I1009 14:54:01.070883 139781364750208 model_lib_v2.py:652] Step 130700 per-step time 0.254s loss=0.715
    INFO:tensorflow:Step 130800 per-step time 0.243s loss=0.800
    I1009 14:54:25.408555 139781364750208 model_lib_v2.py:652] Step 130800 per-step time 0.243s loss=0.800
    INFO:tensorflow:Step 130900 per-step time 0.242s loss=0.922
    I1009 14:54:49.832186 139781364750208 model_lib_v2.py:652] Step 130900 per-step time 0.242s loss=0.922
    INFO:tensorflow:Step 131000 per-step time 0.256s loss=0.770
    I1009 14:55:14.263882 139781364750208 model_lib_v2.py:652] Step 131000 per-step time 0.256s loss=0.770
    INFO:tensorflow:Step 131100 per-step time 0.231s loss=1.012
    I1009 14:55:39.965109 139781364750208 model_lib_v2.py:652] Step 131100 per-step time 0.231s loss=1.012
    INFO:tensorflow:Step 131200 per-step time 0.250s loss=0.447
    I1009 14:56:04.325111 139781364750208 model_lib_v2.py:652] Step 131200 per-step time 0.250s loss=0.447
    INFO:tensorflow:Step 131300 per-step time 0.237s loss=0.524
    I1009 14:56:28.625797 139781364750208 model_lib_v2.py:652] Step 131300 per-step time 0.237s loss=0.524
    INFO:tensorflow:Step 131400 per-step time 0.237s loss=0.740
    I1009 14:56:53.181031 139781364750208 model_lib_v2.py:652] Step 131400 per-step time 0.237s loss=0.740
    INFO:tensorflow:Step 131500 per-step time 0.255s loss=0.829
    I1009 14:57:17.571098 139781364750208 model_lib_v2.py:652] Step 131500 per-step time 0.255s loss=0.829
    INFO:tensorflow:Step 131600 per-step time 0.258s loss=0.846
    I1009 14:57:42.135200 139781364750208 model_lib_v2.py:652] Step 131600 per-step time 0.258s loss=0.846
    INFO:tensorflow:Step 131700 per-step time 0.239s loss=0.834
    I1009 14:58:06.587533 139781364750208 model_lib_v2.py:652] Step 131700 per-step time 0.239s loss=0.834
    INFO:tensorflow:Step 131800 per-step time 0.251s loss=0.594
    I1009 14:58:30.993989 139781364750208 model_lib_v2.py:652] Step 131800 per-step time 0.251s loss=0.594
    INFO:tensorflow:Step 131900 per-step time 0.244s loss=0.556
    I1009 14:58:55.399582 139781364750208 model_lib_v2.py:652] Step 131900 per-step time 0.244s loss=0.556
    INFO:tensorflow:Step 132000 per-step time 0.238s loss=0.779
    I1009 14:59:19.977268 139781364750208 model_lib_v2.py:652] Step 132000 per-step time 0.238s loss=0.779
    INFO:tensorflow:Step 132100 per-step time 0.242s loss=0.510
    I1009 14:59:45.632843 139781364750208 model_lib_v2.py:652] Step 132100 per-step time 0.242s loss=0.510
    INFO:tensorflow:Step 132200 per-step time 0.250s loss=0.659
    I1009 15:00:10.157603 139781364750208 model_lib_v2.py:652] Step 132200 per-step time 0.250s loss=0.659
    INFO:tensorflow:Step 132300 per-step time 0.264s loss=0.717
    I1009 15:00:35.006028 139781364750208 model_lib_v2.py:652] Step 132300 per-step time 0.264s loss=0.717
    INFO:tensorflow:Step 132400 per-step time 0.243s loss=0.556
    I1009 15:00:59.375652 139781364750208 model_lib_v2.py:652] Step 132400 per-step time 0.243s loss=0.556
    INFO:tensorflow:Step 132500 per-step time 0.236s loss=0.654
    I1009 15:01:23.872752 139781364750208 model_lib_v2.py:652] Step 132500 per-step time 0.236s loss=0.654
    INFO:tensorflow:Step 132600 per-step time 0.237s loss=0.887
    I1009 15:01:48.438448 139781364750208 model_lib_v2.py:652] Step 132600 per-step time 0.237s loss=0.887
    INFO:tensorflow:Step 132700 per-step time 0.242s loss=0.957
    I1009 15:02:12.923369 139781364750208 model_lib_v2.py:652] Step 132700 per-step time 0.242s loss=0.957
    INFO:tensorflow:Step 132800 per-step time 0.249s loss=0.577
    I1009 15:02:37.383532 139781364750208 model_lib_v2.py:652] Step 132800 per-step time 0.249s loss=0.577
    INFO:tensorflow:Step 132900 per-step time 0.245s loss=0.478
    I1009 15:03:01.756216 139781364750208 model_lib_v2.py:652] Step 132900 per-step time 0.245s loss=0.478
    INFO:tensorflow:Step 133000 per-step time 0.243s loss=0.551
    I1009 15:03:26.166739 139781364750208 model_lib_v2.py:652] Step 133000 per-step time 0.243s loss=0.551
    INFO:tensorflow:Step 133100 per-step time 0.240s loss=1.148
    I1009 15:03:51.351450 139781364750208 model_lib_v2.py:652] Step 133100 per-step time 0.240s loss=1.148
    INFO:tensorflow:Step 133200 per-step time 0.243s loss=1.195
    I1009 15:04:15.749828 139781364750208 model_lib_v2.py:652] Step 133200 per-step time 0.243s loss=1.195
    INFO:tensorflow:Step 133300 per-step time 0.252s loss=0.572
    I1009 15:04:40.157006 139781364750208 model_lib_v2.py:652] Step 133300 per-step time 0.252s loss=0.572
    INFO:tensorflow:Step 133400 per-step time 0.226s loss=0.869
    I1009 15:05:04.544734 139781364750208 model_lib_v2.py:652] Step 133400 per-step time 0.226s loss=0.869
    INFO:tensorflow:Step 133500 per-step time 0.250s loss=0.611
    I1009 15:05:28.870006 139781364750208 model_lib_v2.py:652] Step 133500 per-step time 0.250s loss=0.611
    INFO:tensorflow:Step 133600 per-step time 0.236s loss=0.813
    I1009 15:05:53.454959 139781364750208 model_lib_v2.py:652] Step 133600 per-step time 0.236s loss=0.813
    INFO:tensorflow:Step 133700 per-step time 0.246s loss=0.578
    I1009 15:06:17.841100 139781364750208 model_lib_v2.py:652] Step 133700 per-step time 0.246s loss=0.578
    INFO:tensorflow:Step 133800 per-step time 0.248s loss=0.732
    I1009 15:06:42.215551 139781364750208 model_lib_v2.py:652] Step 133800 per-step time 0.248s loss=0.732
    INFO:tensorflow:Step 133900 per-step time 0.235s loss=0.473
    I1009 15:07:06.453290 139781364750208 model_lib_v2.py:652] Step 133900 per-step time 0.235s loss=0.473
    INFO:tensorflow:Step 134000 per-step time 0.244s loss=0.845
    I1009 15:07:30.804987 139781364750208 model_lib_v2.py:652] Step 134000 per-step time 0.244s loss=0.845
    INFO:tensorflow:Step 134100 per-step time 0.244s loss=0.407
    I1009 15:07:56.077418 139781364750208 model_lib_v2.py:652] Step 134100 per-step time 0.244s loss=0.407
    INFO:tensorflow:Step 134200 per-step time 0.248s loss=0.605
    I1009 15:08:20.253557 139781364750208 model_lib_v2.py:652] Step 134200 per-step time 0.248s loss=0.605
    INFO:tensorflow:Step 134300 per-step time 0.248s loss=0.762
    I1009 15:08:44.535164 139781364750208 model_lib_v2.py:652] Step 134300 per-step time 0.248s loss=0.762
    INFO:tensorflow:Step 134400 per-step time 0.255s loss=0.753
    I1009 15:09:08.810174 139781364750208 model_lib_v2.py:652] Step 134400 per-step time 0.255s loss=0.753
    INFO:tensorflow:Step 134500 per-step time 0.246s loss=0.480
    I1009 15:09:33.333606 139781364750208 model_lib_v2.py:652] Step 134500 per-step time 0.246s loss=0.480
    INFO:tensorflow:Step 134600 per-step time 0.247s loss=0.820
    I1009 15:09:57.765939 139781364750208 model_lib_v2.py:652] Step 134600 per-step time 0.247s loss=0.820
    INFO:tensorflow:Step 134700 per-step time 0.231s loss=0.655
    I1009 15:10:22.260170 139781364750208 model_lib_v2.py:652] Step 134700 per-step time 0.231s loss=0.655
    INFO:tensorflow:Step 134800 per-step time 0.250s loss=0.774
    I1009 15:10:46.712036 139781364750208 model_lib_v2.py:652] Step 134800 per-step time 0.250s loss=0.774
    INFO:tensorflow:Step 134900 per-step time 0.247s loss=0.904
    I1009 15:11:11.457262 139781364750208 model_lib_v2.py:652] Step 134900 per-step time 0.247s loss=0.904
    INFO:tensorflow:Step 135000 per-step time 0.242s loss=0.919
    I1009 15:11:35.856112 139781364750208 model_lib_v2.py:652] Step 135000 per-step time 0.242s loss=0.919
    INFO:tensorflow:Step 135100 per-step time 0.239s loss=0.633
    I1009 15:12:01.014553 139781364750208 model_lib_v2.py:652] Step 135100 per-step time 0.239s loss=0.633
    INFO:tensorflow:Step 135200 per-step time 0.259s loss=1.106
    I1009 15:12:25.322070 139781364750208 model_lib_v2.py:652] Step 135200 per-step time 0.259s loss=1.106
    INFO:tensorflow:Step 135300 per-step time 0.243s loss=0.475
    I1009 15:12:49.678848 139781364750208 model_lib_v2.py:652] Step 135300 per-step time 0.243s loss=0.475
    INFO:tensorflow:Step 135400 per-step time 0.249s loss=0.880
    I1009 15:13:13.815092 139781364750208 model_lib_v2.py:652] Step 135400 per-step time 0.249s loss=0.880
    INFO:tensorflow:Step 135500 per-step time 0.239s loss=0.810
    I1009 15:13:38.076730 139781364750208 model_lib_v2.py:652] Step 135500 per-step time 0.239s loss=0.810
    INFO:tensorflow:Step 135600 per-step time 0.250s loss=0.608
    I1009 15:14:02.086019 139781364750208 model_lib_v2.py:652] Step 135600 per-step time 0.250s loss=0.608
    INFO:tensorflow:Step 135700 per-step time 0.230s loss=0.963
    I1009 15:14:26.259772 139781364750208 model_lib_v2.py:652] Step 135700 per-step time 0.230s loss=0.963
    INFO:tensorflow:Step 135800 per-step time 0.241s loss=0.846
    I1009 15:14:50.113464 139781364750208 model_lib_v2.py:652] Step 135800 per-step time 0.241s loss=0.846
    INFO:tensorflow:Step 135900 per-step time 0.252s loss=0.924
    I1009 15:15:14.242081 139781364750208 model_lib_v2.py:652] Step 135900 per-step time 0.252s loss=0.924
    INFO:tensorflow:Step 136000 per-step time 0.239s loss=0.792
    I1009 15:15:38.989482 139781364750208 model_lib_v2.py:652] Step 136000 per-step time 0.239s loss=0.792
    INFO:tensorflow:Step 136100 per-step time 0.234s loss=0.465
    I1009 15:16:04.560241 139781364750208 model_lib_v2.py:652] Step 136100 per-step time 0.234s loss=0.465
    INFO:tensorflow:Step 136200 per-step time 0.237s loss=0.648
    I1009 15:16:28.940083 139781364750208 model_lib_v2.py:652] Step 136200 per-step time 0.237s loss=0.648
    INFO:tensorflow:Step 136300 per-step time 0.246s loss=0.570
    I1009 15:16:53.358058 139781364750208 model_lib_v2.py:652] Step 136300 per-step time 0.246s loss=0.570
    INFO:tensorflow:Step 136400 per-step time 0.243s loss=0.699
    I1009 15:17:17.753012 139781364750208 model_lib_v2.py:652] Step 136400 per-step time 0.243s loss=0.699
    INFO:tensorflow:Step 136500 per-step time 0.250s loss=0.519
    I1009 15:17:42.076026 139781364750208 model_lib_v2.py:652] Step 136500 per-step time 0.250s loss=0.519
    INFO:tensorflow:Step 136600 per-step time 0.256s loss=0.928
    I1009 15:18:06.501611 139781364750208 model_lib_v2.py:652] Step 136600 per-step time 0.256s loss=0.928
    INFO:tensorflow:Step 136700 per-step time 0.262s loss=0.606
    I1009 15:18:31.058767 139781364750208 model_lib_v2.py:652] Step 136700 per-step time 0.262s loss=0.606
    INFO:tensorflow:Step 136800 per-step time 0.243s loss=0.741
    I1009 15:18:55.368114 139781364750208 model_lib_v2.py:652] Step 136800 per-step time 0.243s loss=0.741
    INFO:tensorflow:Step 136900 per-step time 0.244s loss=0.488
    I1009 15:19:19.760471 139781364750208 model_lib_v2.py:652] Step 136900 per-step time 0.244s loss=0.488
    INFO:tensorflow:Step 137000 per-step time 0.251s loss=0.694
    I1009 15:19:44.240297 139781364750208 model_lib_v2.py:652] Step 137000 per-step time 0.251s loss=0.694
    INFO:tensorflow:Step 137100 per-step time 0.237s loss=1.173
    I1009 15:20:09.596210 139781364750208 model_lib_v2.py:652] Step 137100 per-step time 0.237s loss=1.173
    INFO:tensorflow:Step 137200 per-step time 0.246s loss=0.524
    I1009 15:20:33.968957 139781364750208 model_lib_v2.py:652] Step 137200 per-step time 0.246s loss=0.524
    INFO:tensorflow:Step 137300 per-step time 0.247s loss=0.505
    I1009 15:20:58.285846 139781364750208 model_lib_v2.py:652] Step 137300 per-step time 0.247s loss=0.505
    INFO:tensorflow:Step 137400 per-step time 0.240s loss=0.876
    I1009 15:21:22.870270 139781364750208 model_lib_v2.py:652] Step 137400 per-step time 0.240s loss=0.876
    INFO:tensorflow:Step 137500 per-step time 0.237s loss=0.506
    I1009 15:21:47.207288 139781364750208 model_lib_v2.py:652] Step 137500 per-step time 0.237s loss=0.506
    INFO:tensorflow:Step 137600 per-step time 0.238s loss=0.752
    I1009 15:22:11.211851 139781364750208 model_lib_v2.py:652] Step 137600 per-step time 0.238s loss=0.752
    INFO:tensorflow:Step 137700 per-step time 0.233s loss=0.554
    I1009 15:22:35.138053 139781364750208 model_lib_v2.py:652] Step 137700 per-step time 0.233s loss=0.554
    INFO:tensorflow:Step 137800 per-step time 0.243s loss=0.901
    I1009 15:22:59.201087 139781364750208 model_lib_v2.py:652] Step 137800 per-step time 0.243s loss=0.901
    INFO:tensorflow:Step 137900 per-step time 0.237s loss=1.057
    I1009 15:23:23.397197 139781364750208 model_lib_v2.py:652] Step 137900 per-step time 0.237s loss=1.057
    INFO:tensorflow:Step 138000 per-step time 0.259s loss=1.178
    I1009 15:23:47.753016 139781364750208 model_lib_v2.py:652] Step 138000 per-step time 0.259s loss=1.178
    INFO:tensorflow:Step 138100 per-step time 0.250s loss=0.553
    I1009 15:24:13.580804 139781364750208 model_lib_v2.py:652] Step 138100 per-step time 0.250s loss=0.553
    INFO:tensorflow:Step 138200 per-step time 0.249s loss=0.680
    I1009 15:24:38.565869 139781364750208 model_lib_v2.py:652] Step 138200 per-step time 0.249s loss=0.680
    INFO:tensorflow:Step 138300 per-step time 0.256s loss=0.411
    I1009 15:25:03.452721 139781364750208 model_lib_v2.py:652] Step 138300 per-step time 0.256s loss=0.411
    INFO:tensorflow:Step 138400 per-step time 0.247s loss=0.647
    I1009 15:25:28.160630 139781364750208 model_lib_v2.py:652] Step 138400 per-step time 0.247s loss=0.647
    INFO:tensorflow:Step 138500 per-step time 0.242s loss=0.739
    I1009 15:25:53.000633 139781364750208 model_lib_v2.py:652] Step 138500 per-step time 0.242s loss=0.739
    INFO:tensorflow:Step 138600 per-step time 0.265s loss=0.805
    I1009 15:26:17.883937 139781364750208 model_lib_v2.py:652] Step 138600 per-step time 0.265s loss=0.805
    INFO:tensorflow:Step 138700 per-step time 0.231s loss=0.783
    I1009 15:26:42.870568 139781364750208 model_lib_v2.py:652] Step 138700 per-step time 0.231s loss=0.783
    INFO:tensorflow:Step 138800 per-step time 0.239s loss=0.576
    I1009 15:27:07.514240 139781364750208 model_lib_v2.py:652] Step 138800 per-step time 0.239s loss=0.576
    INFO:tensorflow:Step 138900 per-step time 0.247s loss=0.673
    I1009 15:27:32.161687 139781364750208 model_lib_v2.py:652] Step 138900 per-step time 0.247s loss=0.673
    INFO:tensorflow:Step 139000 per-step time 0.253s loss=0.623
    I1009 15:27:56.916695 139781364750208 model_lib_v2.py:652] Step 139000 per-step time 0.253s loss=0.623
    INFO:tensorflow:Step 139100 per-step time 0.254s loss=0.645
    I1009 15:28:22.552536 139781364750208 model_lib_v2.py:652] Step 139100 per-step time 0.254s loss=0.645
    INFO:tensorflow:Step 139200 per-step time 0.233s loss=0.727
    I1009 15:28:47.115586 139781364750208 model_lib_v2.py:652] Step 139200 per-step time 0.233s loss=0.727
    INFO:tensorflow:Step 139300 per-step time 0.255s loss=0.693
    I1009 15:29:11.640704 139781364750208 model_lib_v2.py:652] Step 139300 per-step time 0.255s loss=0.693
    INFO:tensorflow:Step 139400 per-step time 0.247s loss=0.754
    I1009 15:29:36.083740 139781364750208 model_lib_v2.py:652] Step 139400 per-step time 0.247s loss=0.754
    INFO:tensorflow:Step 139500 per-step time 0.249s loss=0.646
    I1009 15:30:00.677098 139781364750208 model_lib_v2.py:652] Step 139500 per-step time 0.249s loss=0.646
    INFO:tensorflow:Step 139600 per-step time 0.234s loss=0.742
    I1009 15:30:25.280663 139781364750208 model_lib_v2.py:652] Step 139600 per-step time 0.234s loss=0.742
    INFO:tensorflow:Step 139700 per-step time 0.250s loss=0.858
    I1009 15:30:49.778049 139781364750208 model_lib_v2.py:652] Step 139700 per-step time 0.250s loss=0.858
    INFO:tensorflow:Step 139800 per-step time 0.244s loss=0.544
    I1009 15:31:14.457546 139781364750208 model_lib_v2.py:652] Step 139800 per-step time 0.244s loss=0.544
    INFO:tensorflow:Step 139900 per-step time 0.249s loss=0.965
    I1009 15:31:39.070051 139781364750208 model_lib_v2.py:652] Step 139900 per-step time 0.249s loss=0.965
    INFO:tensorflow:Step 140000 per-step time 0.234s loss=0.888
    I1009 15:32:03.627246 139781364750208 model_lib_v2.py:652] Step 140000 per-step time 0.234s loss=0.888
    INFO:tensorflow:Step 140100 per-step time 0.229s loss=1.127
    I1009 15:32:29.091325 139781364750208 model_lib_v2.py:652] Step 140100 per-step time 0.229s loss=1.127
    INFO:tensorflow:Step 140200 per-step time 0.239s loss=0.648
    I1009 15:32:53.616589 139781364750208 model_lib_v2.py:652] Step 140200 per-step time 0.239s loss=0.648
    INFO:tensorflow:Step 140300 per-step time 0.237s loss=0.645
    I1009 15:33:18.080072 139781364750208 model_lib_v2.py:652] Step 140300 per-step time 0.237s loss=0.645
    INFO:tensorflow:Step 140400 per-step time 0.250s loss=0.758
    I1009 15:33:42.475146 139781364750208 model_lib_v2.py:652] Step 140400 per-step time 0.250s loss=0.758
    INFO:tensorflow:Step 140500 per-step time 0.235s loss=0.850
    I1009 15:34:06.887393 139781364750208 model_lib_v2.py:652] Step 140500 per-step time 0.235s loss=0.850
    INFO:tensorflow:Step 140600 per-step time 0.233s loss=0.699
    I1009 15:34:31.269140 139781364750208 model_lib_v2.py:652] Step 140600 per-step time 0.233s loss=0.699
    INFO:tensorflow:Step 140700 per-step time 0.246s loss=0.961
    I1009 15:34:55.966883 139781364750208 model_lib_v2.py:652] Step 140700 per-step time 0.246s loss=0.961
    INFO:tensorflow:Step 140800 per-step time 0.237s loss=0.833
    I1009 15:35:20.071454 139781364750208 model_lib_v2.py:652] Step 140800 per-step time 0.237s loss=0.833
    INFO:tensorflow:Step 140900 per-step time 0.242s loss=0.708
    I1009 15:35:44.338597 139781364750208 model_lib_v2.py:652] Step 140900 per-step time 0.242s loss=0.708
    INFO:tensorflow:Step 141000 per-step time 0.232s loss=0.590
    I1009 15:36:08.494493 139781364750208 model_lib_v2.py:652] Step 141000 per-step time 0.232s loss=0.590
    INFO:tensorflow:Step 141100 per-step time 0.256s loss=0.810
    I1009 15:36:33.611613 139781364750208 model_lib_v2.py:652] Step 141100 per-step time 0.256s loss=0.810
    INFO:tensorflow:Step 141200 per-step time 0.242s loss=0.804
    I1009 15:36:58.242609 139781364750208 model_lib_v2.py:652] Step 141200 per-step time 0.242s loss=0.804
    INFO:tensorflow:Step 141300 per-step time 0.245s loss=0.895
    I1009 15:37:22.615503 139781364750208 model_lib_v2.py:652] Step 141300 per-step time 0.245s loss=0.895
    INFO:tensorflow:Step 141400 per-step time 0.238s loss=1.250
    I1009 15:37:47.299506 139781364750208 model_lib_v2.py:652] Step 141400 per-step time 0.238s loss=1.250
    INFO:tensorflow:Step 141500 per-step time 0.247s loss=0.557
    I1009 15:38:11.924633 139781364750208 model_lib_v2.py:652] Step 141500 per-step time 0.247s loss=0.557
    INFO:tensorflow:Step 141600 per-step time 0.243s loss=0.788
    I1009 15:38:36.558940 139781364750208 model_lib_v2.py:652] Step 141600 per-step time 0.243s loss=0.788
    INFO:tensorflow:Step 141700 per-step time 0.242s loss=0.785
    I1009 15:39:00.938670 139781364750208 model_lib_v2.py:652] Step 141700 per-step time 0.242s loss=0.785
    INFO:tensorflow:Step 141800 per-step time 0.243s loss=0.480
    I1009 15:39:25.444340 139781364750208 model_lib_v2.py:652] Step 141800 per-step time 0.243s loss=0.480
    INFO:tensorflow:Step 141900 per-step time 0.242s loss=0.571
    I1009 15:39:49.809199 139781364750208 model_lib_v2.py:652] Step 141900 per-step time 0.242s loss=0.571
    INFO:tensorflow:Step 142000 per-step time 0.254s loss=0.677
    I1009 15:40:14.123504 139781364750208 model_lib_v2.py:652] Step 142000 per-step time 0.254s loss=0.677
    INFO:tensorflow:Step 142100 per-step time 0.235s loss=0.909
    I1009 15:40:39.424687 139781364750208 model_lib_v2.py:652] Step 142100 per-step time 0.235s loss=0.909
    INFO:tensorflow:Step 142200 per-step time 0.252s loss=0.736
    I1009 15:41:03.805864 139781364750208 model_lib_v2.py:652] Step 142200 per-step time 0.252s loss=0.736
    INFO:tensorflow:Step 142300 per-step time 0.251s loss=0.806
    I1009 15:41:28.038803 139781364750208 model_lib_v2.py:652] Step 142300 per-step time 0.251s loss=0.806
    INFO:tensorflow:Step 142400 per-step time 0.236s loss=0.611
    I1009 15:41:52.193959 139781364750208 model_lib_v2.py:652] Step 142400 per-step time 0.236s loss=0.611
    INFO:tensorflow:Step 142500 per-step time 0.245s loss=0.650
    I1009 15:42:16.844622 139781364750208 model_lib_v2.py:652] Step 142500 per-step time 0.245s loss=0.650
    INFO:tensorflow:Step 142600 per-step time 0.232s loss=0.519
    I1009 15:42:41.168614 139781364750208 model_lib_v2.py:652] Step 142600 per-step time 0.232s loss=0.519
    INFO:tensorflow:Step 142700 per-step time 0.258s loss=0.925
    I1009 15:43:05.578829 139781364750208 model_lib_v2.py:652] Step 142700 per-step time 0.258s loss=0.925
    INFO:tensorflow:Step 142800 per-step time 0.244s loss=0.662
    I1009 15:43:29.757430 139781364750208 model_lib_v2.py:652] Step 142800 per-step time 0.244s loss=0.662
    INFO:tensorflow:Step 142900 per-step time 0.256s loss=0.778
    I1009 15:43:54.115697 139781364750208 model_lib_v2.py:652] Step 142900 per-step time 0.256s loss=0.778
    INFO:tensorflow:Step 143000 per-step time 0.242s loss=0.660
    I1009 15:44:18.516705 139781364750208 model_lib_v2.py:652] Step 143000 per-step time 0.242s loss=0.660
    INFO:tensorflow:Step 143100 per-step time 0.233s loss=0.843
    I1009 15:44:43.839704 139781364750208 model_lib_v2.py:652] Step 143100 per-step time 0.233s loss=0.843
    INFO:tensorflow:Step 143200 per-step time 0.230s loss=0.549
    I1009 15:45:08.294187 139781364750208 model_lib_v2.py:652] Step 143200 per-step time 0.230s loss=0.549
    INFO:tensorflow:Step 143300 per-step time 0.241s loss=0.684
    I1009 15:45:32.614568 139781364750208 model_lib_v2.py:652] Step 143300 per-step time 0.241s loss=0.684
    INFO:tensorflow:Step 143400 per-step time 0.246s loss=0.762
    I1009 15:45:57.052051 139781364750208 model_lib_v2.py:652] Step 143400 per-step time 0.246s loss=0.762
    INFO:tensorflow:Step 143500 per-step time 0.245s loss=0.580
    I1009 15:46:21.493491 139781364750208 model_lib_v2.py:652] Step 143500 per-step time 0.245s loss=0.580
    INFO:tensorflow:Step 143600 per-step time 0.245s loss=0.774
    I1009 15:46:45.920489 139781364750208 model_lib_v2.py:652] Step 143600 per-step time 0.245s loss=0.774
    INFO:tensorflow:Step 143700 per-step time 0.245s loss=0.726
    I1009 15:47:10.422899 139781364750208 model_lib_v2.py:652] Step 143700 per-step time 0.245s loss=0.726
    INFO:tensorflow:Step 143800 per-step time 0.248s loss=0.737
    I1009 15:47:34.811802 139781364750208 model_lib_v2.py:652] Step 143800 per-step time 0.248s loss=0.737
    INFO:tensorflow:Step 143900 per-step time 0.250s loss=0.733
    I1009 15:47:59.122415 139781364750208 model_lib_v2.py:652] Step 143900 per-step time 0.250s loss=0.733
    INFO:tensorflow:Step 144000 per-step time 0.259s loss=0.739
    I1009 15:48:23.367782 139781364750208 model_lib_v2.py:652] Step 144000 per-step time 0.259s loss=0.739
    INFO:tensorflow:Step 144100 per-step time 0.247s loss=0.665
    I1009 15:48:48.649172 139781364750208 model_lib_v2.py:652] Step 144100 per-step time 0.247s loss=0.665
    INFO:tensorflow:Step 144200 per-step time 0.248s loss=0.548
    I1009 15:49:13.002673 139781364750208 model_lib_v2.py:652] Step 144200 per-step time 0.248s loss=0.548
    INFO:tensorflow:Step 144300 per-step time 0.251s loss=0.540
    I1009 15:49:37.350743 139781364750208 model_lib_v2.py:652] Step 144300 per-step time 0.251s loss=0.540
    INFO:tensorflow:Step 144400 per-step time 0.248s loss=0.585
    I1009 15:50:01.683957 139781364750208 model_lib_v2.py:652] Step 144400 per-step time 0.248s loss=0.585
    INFO:tensorflow:Step 144500 per-step time 0.244s loss=0.770
    I1009 15:50:26.009359 139781364750208 model_lib_v2.py:652] Step 144500 per-step time 0.244s loss=0.770
    INFO:tensorflow:Step 144600 per-step time 0.252s loss=0.728
    I1009 15:50:50.181319 139781364750208 model_lib_v2.py:652] Step 144600 per-step time 0.252s loss=0.728
    INFO:tensorflow:Step 144700 per-step time 0.241s loss=0.638
    I1009 15:51:14.470136 139781364750208 model_lib_v2.py:652] Step 144700 per-step time 0.241s loss=0.638
    INFO:tensorflow:Step 144800 per-step time 0.245s loss=0.867
    I1009 15:51:38.707280 139781364750208 model_lib_v2.py:652] Step 144800 per-step time 0.245s loss=0.867
    INFO:tensorflow:Step 144900 per-step time 0.250s loss=0.702
    I1009 15:52:03.081729 139781364750208 model_lib_v2.py:652] Step 144900 per-step time 0.250s loss=0.702
    INFO:tensorflow:Step 145000 per-step time 0.238s loss=0.740
    I1009 15:52:27.662001 139781364750208 model_lib_v2.py:652] Step 145000 per-step time 0.238s loss=0.740
    INFO:tensorflow:Step 145100 per-step time 0.239s loss=0.579
    I1009 15:52:53.156645 139781364750208 model_lib_v2.py:652] Step 145100 per-step time 0.239s loss=0.579
    INFO:tensorflow:Step 145200 per-step time 0.247s loss=0.963
    I1009 15:53:17.591398 139781364750208 model_lib_v2.py:652] Step 145200 per-step time 0.247s loss=0.963
    INFO:tensorflow:Step 145300 per-step time 0.252s loss=0.418
    I1009 15:53:41.975146 139781364750208 model_lib_v2.py:652] Step 145300 per-step time 0.252s loss=0.418
    INFO:tensorflow:Step 145400 per-step time 0.243s loss=0.679
    I1009 15:54:06.396735 139781364750208 model_lib_v2.py:652] Step 145400 per-step time 0.243s loss=0.679
    INFO:tensorflow:Step 145500 per-step time 0.248s loss=0.509
    I1009 15:54:30.733165 139781364750208 model_lib_v2.py:652] Step 145500 per-step time 0.248s loss=0.509
    INFO:tensorflow:Step 145600 per-step time 0.244s loss=0.596
    I1009 15:54:55.163152 139781364750208 model_lib_v2.py:652] Step 145600 per-step time 0.244s loss=0.596
    INFO:tensorflow:Step 145700 per-step time 0.250s loss=0.733
    I1009 15:55:19.754964 139781364750208 model_lib_v2.py:652] Step 145700 per-step time 0.250s loss=0.733
    INFO:tensorflow:Step 145800 per-step time 0.247s loss=0.906
    I1009 15:55:44.120501 139781364750208 model_lib_v2.py:652] Step 145800 per-step time 0.247s loss=0.906
    INFO:tensorflow:Step 145900 per-step time 0.246s loss=0.751
    I1009 15:56:08.290109 139781364750208 model_lib_v2.py:652] Step 145900 per-step time 0.246s loss=0.751
    INFO:tensorflow:Step 146000 per-step time 0.237s loss=0.625
    I1009 15:56:32.531179 139781364750208 model_lib_v2.py:652] Step 146000 per-step time 0.237s loss=0.625
    INFO:tensorflow:Step 146100 per-step time 0.233s loss=0.542
    I1009 15:56:57.538583 139781364750208 model_lib_v2.py:652] Step 146100 per-step time 0.233s loss=0.542
    INFO:tensorflow:Step 146200 per-step time 0.231s loss=1.136
    I1009 15:57:21.700264 139781364750208 model_lib_v2.py:652] Step 146200 per-step time 0.231s loss=1.136
    INFO:tensorflow:Step 146300 per-step time 0.245s loss=0.619
    I1009 15:57:46.030875 139781364750208 model_lib_v2.py:652] Step 146300 per-step time 0.245s loss=0.619
    INFO:tensorflow:Step 146400 per-step time 0.235s loss=1.018
    I1009 15:58:10.241083 139781364750208 model_lib_v2.py:652] Step 146400 per-step time 0.235s loss=1.018
    INFO:tensorflow:Step 146500 per-step time 0.229s loss=0.574
    I1009 15:58:34.349671 139781364750208 model_lib_v2.py:652] Step 146500 per-step time 0.229s loss=0.574
    INFO:tensorflow:Step 146600 per-step time 0.237s loss=0.560
    I1009 15:58:58.471029 139781364750208 model_lib_v2.py:652] Step 146600 per-step time 0.237s loss=0.560
    INFO:tensorflow:Step 146700 per-step time 0.230s loss=0.723
    I1009 15:59:22.535205 139781364750208 model_lib_v2.py:652] Step 146700 per-step time 0.230s loss=0.723
    INFO:tensorflow:Step 146800 per-step time 0.243s loss=0.650
    I1009 15:59:46.653309 139781364750208 model_lib_v2.py:652] Step 146800 per-step time 0.243s loss=0.650
    INFO:tensorflow:Step 146900 per-step time 0.231s loss=0.564
    I1009 16:00:10.592025 139781364750208 model_lib_v2.py:652] Step 146900 per-step time 0.231s loss=0.564
    INFO:tensorflow:Step 147000 per-step time 0.244s loss=0.546
    I1009 16:00:34.736291 139781364750208 model_lib_v2.py:652] Step 147000 per-step time 0.244s loss=0.546
    INFO:tensorflow:Step 147100 per-step time 0.235s loss=0.653
    I1009 16:00:59.823195 139781364750208 model_lib_v2.py:652] Step 147100 per-step time 0.235s loss=0.653
    INFO:tensorflow:Step 147200 per-step time 0.245s loss=0.583
    I1009 16:01:23.957996 139781364750208 model_lib_v2.py:652] Step 147200 per-step time 0.245s loss=0.583
    INFO:tensorflow:Step 147300 per-step time 0.239s loss=0.866
    I1009 16:01:48.117817 139781364750208 model_lib_v2.py:652] Step 147300 per-step time 0.239s loss=0.866
    INFO:tensorflow:Step 147400 per-step time 0.251s loss=0.723
    I1009 16:02:12.341778 139781364750208 model_lib_v2.py:652] Step 147400 per-step time 0.251s loss=0.723
    INFO:tensorflow:Step 147500 per-step time 0.238s loss=0.849
    I1009 16:02:36.515494 139781364750208 model_lib_v2.py:652] Step 147500 per-step time 0.238s loss=0.849
    INFO:tensorflow:Step 147600 per-step time 0.238s loss=0.878
    I1009 16:03:00.866611 139781364750208 model_lib_v2.py:652] Step 147600 per-step time 0.238s loss=0.878
    INFO:tensorflow:Step 147700 per-step time 0.235s loss=0.734
    I1009 16:03:24.941560 139781364750208 model_lib_v2.py:652] Step 147700 per-step time 0.235s loss=0.734
    INFO:tensorflow:Step 147800 per-step time 0.230s loss=0.796
    I1009 16:03:48.963276 139781364750208 model_lib_v2.py:652] Step 147800 per-step time 0.230s loss=0.796
    INFO:tensorflow:Step 147900 per-step time 0.243s loss=0.956
    I1009 16:04:13.027511 139781364750208 model_lib_v2.py:652] Step 147900 per-step time 0.243s loss=0.956
    INFO:tensorflow:Step 148000 per-step time 0.234s loss=0.754
    I1009 16:04:37.299832 139781364750208 model_lib_v2.py:652] Step 148000 per-step time 0.234s loss=0.754
    INFO:tensorflow:Step 148100 per-step time 0.253s loss=0.479
    I1009 16:05:02.458223 139781364750208 model_lib_v2.py:652] Step 148100 per-step time 0.253s loss=0.479
    INFO:tensorflow:Step 148200 per-step time 0.250s loss=0.517
    I1009 16:05:26.609742 139781364750208 model_lib_v2.py:652] Step 148200 per-step time 0.250s loss=0.517
    INFO:tensorflow:Step 148300 per-step time 0.229s loss=0.758
    I1009 16:05:50.741318 139781364750208 model_lib_v2.py:652] Step 148300 per-step time 0.229s loss=0.758
    INFO:tensorflow:Step 148400 per-step time 0.228s loss=0.586
    I1009 16:06:14.821302 139781364750208 model_lib_v2.py:652] Step 148400 per-step time 0.228s loss=0.586
    INFO:tensorflow:Step 148500 per-step time 0.242s loss=0.613
    I1009 16:06:39.007535 139781364750208 model_lib_v2.py:652] Step 148500 per-step time 0.242s loss=0.613
    INFO:tensorflow:Step 148600 per-step time 0.248s loss=1.030
    I1009 16:07:02.986436 139781364750208 model_lib_v2.py:652] Step 148600 per-step time 0.248s loss=1.030
    INFO:tensorflow:Step 148700 per-step time 0.250s loss=0.716
    I1009 16:07:27.112155 139781364750208 model_lib_v2.py:652] Step 148700 per-step time 0.250s loss=0.716
    INFO:tensorflow:Step 148800 per-step time 0.254s loss=0.531
    I1009 16:07:51.364436 139781364750208 model_lib_v2.py:652] Step 148800 per-step time 0.254s loss=0.531
    INFO:tensorflow:Step 148900 per-step time 0.243s loss=0.625
    I1009 16:08:15.751339 139781364750208 model_lib_v2.py:652] Step 148900 per-step time 0.243s loss=0.625
    INFO:tensorflow:Step 149000 per-step time 0.251s loss=0.568
    I1009 16:08:39.898613 139781364750208 model_lib_v2.py:652] Step 149000 per-step time 0.251s loss=0.568
    INFO:tensorflow:Step 149100 per-step time 0.244s loss=0.878
    I1009 16:09:04.784794 139781364750208 model_lib_v2.py:652] Step 149100 per-step time 0.244s loss=0.878
    INFO:tensorflow:Step 149200 per-step time 0.240s loss=0.800
    I1009 16:09:28.868625 139781364750208 model_lib_v2.py:652] Step 149200 per-step time 0.240s loss=0.800
    INFO:tensorflow:Step 149300 per-step time 0.238s loss=0.666
    I1009 16:09:52.824218 139781364750208 model_lib_v2.py:652] Step 149300 per-step time 0.238s loss=0.666
    INFO:tensorflow:Step 149400 per-step time 0.247s loss=0.869
    I1009 16:10:16.792298 139781364750208 model_lib_v2.py:652] Step 149400 per-step time 0.247s loss=0.869
    INFO:tensorflow:Step 149500 per-step time 0.236s loss=0.636
    I1009 16:10:40.761886 139781364750208 model_lib_v2.py:652] Step 149500 per-step time 0.236s loss=0.636
    INFO:tensorflow:Step 149600 per-step time 0.246s loss=0.770
    I1009 16:11:04.655763 139781364750208 model_lib_v2.py:652] Step 149600 per-step time 0.246s loss=0.770
    INFO:tensorflow:Step 149700 per-step time 0.254s loss=0.721
    I1009 16:11:28.583266 139781364750208 model_lib_v2.py:652] Step 149700 per-step time 0.254s loss=0.721
    INFO:tensorflow:Step 149800 per-step time 0.238s loss=0.630
    I1009 16:11:52.705964 139781364750208 model_lib_v2.py:652] Step 149800 per-step time 0.238s loss=0.630
    INFO:tensorflow:Step 149900 per-step time 0.237s loss=0.618
    I1009 16:12:16.621191 139781364750208 model_lib_v2.py:652] Step 149900 per-step time 0.237s loss=0.618
    INFO:tensorflow:Step 150000 per-step time 0.240s loss=1.027
    I1009 16:12:40.588186 139781364750208 model_lib_v2.py:652] Step 150000 per-step time 0.240s loss=1.027


# Make sure to save the important files to continue training later
    - pipeline config needs to be the same or training fails
    - label_map.pbtxt needs to be the same when training the model again or objects will have the wrong labels if label map is different.


```python
!zip myoutputmodel_150k.zip -r ./myoutputmodel
!cp myoutputmodel_150k.zip drive/My\ Drive/
!rm drive/My\ Drive/pipeline_file.config
!cp pipeline_file.config drive/My\ Drive/
!cp label_map.pbtxt drive/My\ Drive/
```

      adding: myoutputmodel/ (stored 0%)
      adding: myoutputmodel/ckpt-145.data-00000-of-00001 (deflated 11%)
      adding: myoutputmodel/ckpt-149.data-00000-of-00001 (deflated 11%)
      adding: myoutputmodel/ckpt-150.index (deflated 81%)
      adding: myoutputmodel/ckpt-147.index (deflated 81%)
      adding: myoutputmodel/ckpt-149.index (deflated 81%)
      adding: myoutputmodel/ckpt-146.data-00000-of-00001 (deflated 11%)
      adding: myoutputmodel/train/ (stored 0%)
      adding: myoutputmodel/train/events.out.tfevents.1602222489.572f21843685.3333.1504.v2 (deflated 6%)
      adding: myoutputmodel/ckpt-146.index (deflated 81%)
      adding: myoutputmodel/ckpt-151.data-00000-of-00001 (deflated 11%)
      adding: myoutputmodel/ckpt-151.index (deflated 81%)
      adding: myoutputmodel/ckpt-148.index (deflated 81%)
      adding: myoutputmodel/ckpt-147.data-00000-of-00001 (deflated 11%)
      adding: myoutputmodel/ckpt-150.data-00000-of-00001 (deflated 11%)
      adding: myoutputmodel/checkpoint (deflated 76%)
      adding: myoutputmodel/ckpt-148.data-00000-of-00001 (deflated 11%)
      adding: myoutputmodel/ckpt-145.index (deflated 81%)



```python
!cp test.record drive/My\ Drive/
!cp train.record drive/My\ Drive/
```

# Perform evaluation using coco_detection_metric and batch_size of 1. Larger batch size causes memory issues.

NOTE: Specifying the model_dir argument tells tf to evaluate instead of train.

```
eval_config: {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
  batch_size: 1;
}
```




```python
!python models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={'./pipeline_file.config'} \
    --model_dir={'./myoutputmodel'} \
    --checkpoint_dir={'./myoutputmodel'} \
    --alsologtostderr
```

    2020-10-09 17:27:18.792915: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    WARNING:tensorflow:Forced number of epochs for all eval validations to be 1.
    W1009 17:27:26.832923 140592774539136 model_lib_v2.py:925] Forced number of epochs for all eval validations to be 1.
    INFO:tensorflow:Maybe overwriting sample_1_of_n_eval_examples: None
    I1009 17:27:26.833167 140592774539136 config_util.py:552] Maybe overwriting sample_1_of_n_eval_examples: None
    INFO:tensorflow:Maybe overwriting use_bfloat16: False
    I1009 17:27:26.833245 140592774539136 config_util.py:552] Maybe overwriting use_bfloat16: False
    INFO:tensorflow:Maybe overwriting eval_num_epochs: 1
    I1009 17:27:26.833319 140592774539136 config_util.py:552] Maybe overwriting eval_num_epochs: 1
    WARNING:tensorflow:Expected number of evaluation epochs is 1, but instead encountered `eval_on_train_input_config.num_epochs` = 0. Overwriting `num_epochs` to 1.
    W1009 17:27:26.833446 140592774539136 model_lib_v2.py:940] Expected number of evaluation epochs is 1, but instead encountered `eval_on_train_input_config.num_epochs` = 0. Overwriting `num_epochs` to 1.
    2020-10-09 17:27:26.915680: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
    2020-10-09 17:27:26.959065: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 17:27:26.959732: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
    coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.75GiB deviceMemoryBandwidth: 836.37GiB/s
    2020-10-09 17:27:26.959805: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-09 17:27:27.208065: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2020-10-09 17:27:27.355678: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
    2020-10-09 17:27:27.381959: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
    2020-10-09 17:27:27.692006: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
    2020-10-09 17:27:27.727371: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
    2020-10-09 17:27:28.298475: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    2020-10-09 17:27:28.298719: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 17:27:28.299383: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 17:27:28.299942: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
    2020-10-09 17:27:28.300708: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX512F
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2020-10-09 17:27:28.340298: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2000165000 Hz
    2020-10-09 17:27:28.340904: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x185ef40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-10-09 17:27:28.340947: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-10-09 17:27:28.492537: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 17:27:28.493237: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x185f100 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2020-10-09 17:27:28.493270: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
    2020-10-09 17:27:28.498139: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 17:27:28.498780: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
    coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.75GiB deviceMemoryBandwidth: 836.37GiB/s
    2020-10-09 17:27:28.498831: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-09 17:27:28.498889: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2020-10-09 17:27:28.498907: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
    2020-10-09 17:27:28.498926: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
    2020-10-09 17:27:28.498942: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
    2020-10-09 17:27:28.498957: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
    2020-10-09 17:27:28.498973: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    2020-10-09 17:27:28.499048: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 17:27:28.499581: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 17:27:28.500067: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
    2020-10-09 17:27:28.506429: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-09 17:27:32.404099: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-10-09 17:27:32.404183: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0 
    2020-10-09 17:27:32.404212: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N 
    2020-10-09 17:27:32.410020: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 17:27:32.410786: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-09 17:27:32.411327: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
    2020-10-09 17:27:32.411404: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 9621 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0)
    INFO:tensorflow:Reading unweighted datasets: ['test.record']
    I1009 17:27:32.516893 140592774539136 dataset_builder.py:148] Reading unweighted datasets: ['test.record']
    INFO:tensorflow:Reading record datasets for input file: ['test.record']
    I1009 17:27:32.521953 140592774539136 dataset_builder.py:77] Reading record datasets for input file: ['test.record']
    INFO:tensorflow:Number of filenames to read: 1
    I1009 17:27:32.522163 140592774539136 dataset_builder.py:78] Number of filenames to read: 1
    WARNING:tensorflow:num_readers has been reduced to 1 to match input file shards.
    W1009 17:27:32.522304 140592774539136 dataset_builder.py:86] num_readers has been reduced to 1 to match input file shards.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:103: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.experimental_deterministic`.
    W1009 17:27:32.533631 140592774539136 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:103: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.experimental_deterministic`.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:222: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.map()
    W1009 17:27:32.604325 140592774539136 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:222: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.map()
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
    W1009 17:27:36.638031 140592774539136 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/inputs.py:259: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    W1009 17:27:38.145683 140592774539136 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/inputs.py:259: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    INFO:tensorflow:Waiting for new checkpoint at ./myoutputmodel
    I1009 17:27:41.026228 140592774539136 checkpoint_utils.py:125] Waiting for new checkpoint at ./myoutputmodel
    INFO:tensorflow:Found new checkpoint at ./myoutputmodel/ckpt-151
    I1009 17:27:41.030318 140592774539136 checkpoint_utils.py:134] Found new checkpoint at ./myoutputmodel/ckpt-151
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib_v2.py:702: set_learning_phase (from tensorflow.python.keras.backend) is deprecated and will be removed after 2020-10-11.
    Instructions for updating:
    Simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
    W1009 17:27:41.108573 140592774539136 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib_v2.py:702: set_learning_phase (from tensorflow.python.keras.backend) is deprecated and will be removed after 2020-10-11.
    Instructions for updating:
    Simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
    INFO:tensorflow:depth of additional conv before box predictor: 0
    I1009 17:27:49.278048 140592774539136 convolutional_keras_box_predictor.py:154] depth of additional conv before box predictor: 0
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/model_util.py:57: Tensor.experimental_ref (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use ref() instead.
    W1009 17:27:56.704976 140592774539136 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/model_util.py:57: Tensor.experimental_ref (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use ref() instead.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    W1009 17:28:03.570195 140592774539136 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/eval_util.py:878: to_int64 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    W1009 17:28:09.649981 140592774539136 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/eval_util.py:878: to_int64 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    2020-10-09 17:28:16.600575: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2020-10-09 17:28:18.139389: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    INFO:tensorflow:Finished eval step 0
    I1009 17:28:23.902426 140592774539136 model_lib_v2.py:799] Finished eval step 0
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/visualization_utils.py:617: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    tf.py_func is deprecated in TF V2. Instead, there are two
        options available in V2.
        - tf.py_function takes a python function which manipulates tf eager
        tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
        an ndarray (just call tensor.numpy()) but having access to eager tensors
        means `tf.py_function`s can use accelerators such as GPUs as well as
        being differentiable using a gradient tape.
        - tf.numpy_function maintains the semantics of the deprecated tf.py_func
        (it is not differentiable, and manipulates numpy arrays). It drops the
        stateful argument making all functions stateful.
        
    W1009 17:28:24.289992 140592774539136 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/visualization_utils.py:617: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    tf.py_func is deprecated in TF V2. Instead, there are two
        options available in V2.
        - tf.py_function takes a python function which manipulates tf eager
        tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
        an ndarray (just call tensor.numpy()) but having access to eager tensors
        means `tf.py_function`s can use accelerators such as GPUs as well as
        being differentiable using a gradient tape.
        - tf.numpy_function maintains the semantics of the deprecated tf.py_func
        (it is not differentiable, and manipulates numpy arrays). It drops the
        stateful argument making all functions stateful.
        
    INFO:tensorflow:Finished eval step 100
    I1009 17:28:35.096394 140592774539136 model_lib_v2.py:799] Finished eval step 100
    INFO:tensorflow:Finished eval step 200
    I1009 17:28:42.344294 140592774539136 model_lib_v2.py:799] Finished eval step 200
    INFO:tensorflow:Finished eval step 300
    I1009 17:28:49.365666 140592774539136 model_lib_v2.py:799] Finished eval step 300
    INFO:tensorflow:Finished eval step 400
    I1009 17:28:56.577271 140592774539136 model_lib_v2.py:799] Finished eval step 400
    INFO:tensorflow:Finished eval step 500
    I1009 17:29:03.583415 140592774539136 model_lib_v2.py:799] Finished eval step 500
    INFO:tensorflow:Finished eval step 600
    I1009 17:29:11.006297 140592774539136 model_lib_v2.py:799] Finished eval step 600
    INFO:tensorflow:Finished eval step 700
    I1009 17:29:18.645378 140592774539136 model_lib_v2.py:799] Finished eval step 700
    INFO:tensorflow:Finished eval step 800
    I1009 17:29:25.983726 140592774539136 model_lib_v2.py:799] Finished eval step 800
    INFO:tensorflow:Finished eval step 900
    I1009 17:29:33.340496 140592774539136 model_lib_v2.py:799] Finished eval step 900
    INFO:tensorflow:Finished eval step 1000
    I1009 17:29:40.879973 140592774539136 model_lib_v2.py:799] Finished eval step 1000
    INFO:tensorflow:Finished eval step 1100
    I1009 17:29:48.239277 140592774539136 model_lib_v2.py:799] Finished eval step 1100
    INFO:tensorflow:Finished eval step 1200
    I1009 17:29:55.610780 140592774539136 model_lib_v2.py:799] Finished eval step 1200
    INFO:tensorflow:Finished eval step 1300
    I1009 17:30:02.979687 140592774539136 model_lib_v2.py:799] Finished eval step 1300
    INFO:tensorflow:Finished eval step 1400
    I1009 17:30:10.200150 140592774539136 model_lib_v2.py:799] Finished eval step 1400
    INFO:tensorflow:Finished eval step 1500
    I1009 17:30:17.852069 140592774539136 model_lib_v2.py:799] Finished eval step 1500
    INFO:tensorflow:Finished eval step 1600
    I1009 17:30:25.151397 140592774539136 model_lib_v2.py:799] Finished eval step 1600
    INFO:tensorflow:Finished eval step 1700
    I1009 17:30:32.476388 140592774539136 model_lib_v2.py:799] Finished eval step 1700
    INFO:tensorflow:Finished eval step 1800
    I1009 17:30:39.727209 140592774539136 model_lib_v2.py:799] Finished eval step 1800
    INFO:tensorflow:Finished eval step 1900
    I1009 17:30:47.121238 140592774539136 model_lib_v2.py:799] Finished eval step 1900
    INFO:tensorflow:Finished eval step 2000
    I1009 17:30:55.324195 140592774539136 model_lib_v2.py:799] Finished eval step 2000
    INFO:tensorflow:Finished eval step 2100
    I1009 17:31:02.727521 140592774539136 model_lib_v2.py:799] Finished eval step 2100
    INFO:tensorflow:Finished eval step 2200
    I1009 17:31:10.056634 140592774539136 model_lib_v2.py:799] Finished eval step 2200
    INFO:tensorflow:Finished eval step 2300
    I1009 17:31:17.448966 140592774539136 model_lib_v2.py:799] Finished eval step 2300
    INFO:tensorflow:Finished eval step 2400
    I1009 17:31:24.844074 140592774539136 model_lib_v2.py:799] Finished eval step 2400
    INFO:tensorflow:Finished eval step 2500
    I1009 17:31:32.073093 140592774539136 model_lib_v2.py:799] Finished eval step 2500
    INFO:tensorflow:Finished eval step 2600
    I1009 17:31:39.775947 140592774539136 model_lib_v2.py:799] Finished eval step 2600
    INFO:tensorflow:Finished eval step 2700
    I1009 17:31:47.249835 140592774539136 model_lib_v2.py:799] Finished eval step 2700
    INFO:tensorflow:Finished eval step 2800
    I1009 17:31:54.570342 140592774539136 model_lib_v2.py:799] Finished eval step 2800
    INFO:tensorflow:Finished eval step 2900
    I1009 17:32:01.928043 140592774539136 model_lib_v2.py:799] Finished eval step 2900
    INFO:tensorflow:Finished eval step 3000
    I1009 17:32:09.184693 140592774539136 model_lib_v2.py:799] Finished eval step 3000
    INFO:tensorflow:Finished eval step 3100
    I1009 17:32:16.467808 140592774539136 model_lib_v2.py:799] Finished eval step 3100
    INFO:tensorflow:Finished eval step 3200
    I1009 17:32:23.809886 140592774539136 model_lib_v2.py:799] Finished eval step 3200
    INFO:tensorflow:Finished eval step 3300
    I1009 17:32:31.131721 140592774539136 model_lib_v2.py:799] Finished eval step 3300
    INFO:tensorflow:Finished eval step 3400
    I1009 17:32:39.004611 140592774539136 model_lib_v2.py:799] Finished eval step 3400
    INFO:tensorflow:Finished eval step 3500
    I1009 17:32:46.363040 140592774539136 model_lib_v2.py:799] Finished eval step 3500
    INFO:tensorflow:Finished eval step 3600
    I1009 17:32:53.811934 140592774539136 model_lib_v2.py:799] Finished eval step 3600
    INFO:tensorflow:Finished eval step 3700
    I1009 17:33:01.187784 140592774539136 model_lib_v2.py:799] Finished eval step 3700
    INFO:tensorflow:Finished eval step 3800
    I1009 17:33:08.548061 140592774539136 model_lib_v2.py:799] Finished eval step 3800
    INFO:tensorflow:Finished eval step 3900
    I1009 17:33:16.007394 140592774539136 model_lib_v2.py:799] Finished eval step 3900
    INFO:tensorflow:Finished eval step 4000
    I1009 17:33:23.331708 140592774539136 model_lib_v2.py:799] Finished eval step 4000
    INFO:tensorflow:Finished eval step 4100
    I1009 17:33:30.726610 140592774539136 model_lib_v2.py:799] Finished eval step 4100
    INFO:tensorflow:Finished eval step 4200
    I1009 17:33:38.011436 140592774539136 model_lib_v2.py:799] Finished eval step 4200
    INFO:tensorflow:Finished eval step 4300
    I1009 17:33:45.476071 140592774539136 model_lib_v2.py:799] Finished eval step 4300
    INFO:tensorflow:Finished eval step 4400
    I1009 17:33:53.387930 140592774539136 model_lib_v2.py:799] Finished eval step 4400
    INFO:tensorflow:Finished eval step 4500
    I1009 17:34:00.604255 140592774539136 model_lib_v2.py:799] Finished eval step 4500
    INFO:tensorflow:Finished eval step 4600
    I1009 17:34:07.508977 140592774539136 model_lib_v2.py:799] Finished eval step 4600
    INFO:tensorflow:Finished eval step 4700
    I1009 17:34:14.443103 140592774539136 model_lib_v2.py:799] Finished eval step 4700
    INFO:tensorflow:Finished eval step 4800
    I1009 17:34:21.393179 140592774539136 model_lib_v2.py:799] Finished eval step 4800
    INFO:tensorflow:Finished eval step 4900
    I1009 17:34:28.355288 140592774539136 model_lib_v2.py:799] Finished eval step 4900
    INFO:tensorflow:Finished eval step 5000
    I1009 17:34:35.233256 140592774539136 model_lib_v2.py:799] Finished eval step 5000
    INFO:tensorflow:Finished eval step 5100
    I1009 17:34:42.170551 140592774539136 model_lib_v2.py:799] Finished eval step 5100
    INFO:tensorflow:Finished eval step 5200
    I1009 17:34:49.236609 140592774539136 model_lib_v2.py:799] Finished eval step 5200
    INFO:tensorflow:Finished eval step 5300
    I1009 17:34:56.139823 140592774539136 model_lib_v2.py:799] Finished eval step 5300
    INFO:tensorflow:Finished eval step 5400
    I1009 17:35:03.254125 140592774539136 model_lib_v2.py:799] Finished eval step 5400
    INFO:tensorflow:Finished eval step 5500
    I1009 17:35:10.331524 140592774539136 model_lib_v2.py:799] Finished eval step 5500
    INFO:tensorflow:Finished eval step 5600
    I1009 17:35:17.327081 140592774539136 model_lib_v2.py:799] Finished eval step 5600
    INFO:tensorflow:Finished eval step 5700
    I1009 17:35:24.935529 140592774539136 model_lib_v2.py:799] Finished eval step 5700
    INFO:tensorflow:Finished eval step 5800
    I1009 17:35:31.876066 140592774539136 model_lib_v2.py:799] Finished eval step 5800
    INFO:tensorflow:Finished eval step 5900
    I1009 17:35:38.819671 140592774539136 model_lib_v2.py:799] Finished eval step 5900
    INFO:tensorflow:Finished eval step 6000
    I1009 17:35:45.738094 140592774539136 model_lib_v2.py:799] Finished eval step 6000
    INFO:tensorflow:Finished eval step 6100
    I1009 17:35:52.694082 140592774539136 model_lib_v2.py:799] Finished eval step 6100
    INFO:tensorflow:Finished eval step 6200
    I1009 17:35:59.788870 140592774539136 model_lib_v2.py:799] Finished eval step 6200
    INFO:tensorflow:Finished eval step 6300
    I1009 17:36:07.207436 140592774539136 model_lib_v2.py:799] Finished eval step 6300
    INFO:tensorflow:Finished eval step 6400
    I1009 17:36:14.171839 140592774539136 model_lib_v2.py:799] Finished eval step 6400
    INFO:tensorflow:Finished eval step 6500
    I1009 17:36:21.147045 140592774539136 model_lib_v2.py:799] Finished eval step 6500
    INFO:tensorflow:Finished eval step 6600
    I1009 17:36:28.033284 140592774539136 model_lib_v2.py:799] Finished eval step 6600
    INFO:tensorflow:Finished eval step 6700
    I1009 17:36:35.304190 140592774539136 model_lib_v2.py:799] Finished eval step 6700
    INFO:tensorflow:Finished eval step 6800
    I1009 17:36:42.542506 140592774539136 model_lib_v2.py:799] Finished eval step 6800
    INFO:tensorflow:Finished eval step 6900
    I1009 17:36:49.938171 140592774539136 model_lib_v2.py:799] Finished eval step 6900
    INFO:tensorflow:Finished eval step 7000
    I1009 17:36:57.146872 140592774539136 model_lib_v2.py:799] Finished eval step 7000
    INFO:tensorflow:Finished eval step 7100
    I1009 17:37:04.381063 140592774539136 model_lib_v2.py:799] Finished eval step 7100
    INFO:tensorflow:Finished eval step 7200
    I1009 17:37:12.384759 140592774539136 model_lib_v2.py:799] Finished eval step 7200
    INFO:tensorflow:Finished eval step 7300
    I1009 17:37:19.630635 140592774539136 model_lib_v2.py:799] Finished eval step 7300
    INFO:tensorflow:Finished eval step 7400
    I1009 17:37:26.875008 140592774539136 model_lib_v2.py:799] Finished eval step 7400
    INFO:tensorflow:Finished eval step 7500
    I1009 17:37:34.079522 140592774539136 model_lib_v2.py:799] Finished eval step 7500
    INFO:tensorflow:Finished eval step 7600
    I1009 17:37:41.246899 140592774539136 model_lib_v2.py:799] Finished eval step 7600
    INFO:tensorflow:Finished eval step 7700
    I1009 17:37:48.456150 140592774539136 model_lib_v2.py:799] Finished eval step 7700
    INFO:tensorflow:Finished eval step 7800
    I1009 17:37:55.652114 140592774539136 model_lib_v2.py:799] Finished eval step 7800
    INFO:tensorflow:Finished eval step 7900
    I1009 17:38:02.839061 140592774539136 model_lib_v2.py:799] Finished eval step 7900
    INFO:tensorflow:Finished eval step 8000
    I1009 17:38:10.037131 140592774539136 model_lib_v2.py:799] Finished eval step 8000
    INFO:tensorflow:Finished eval step 8100
    I1009 17:38:17.255574 140592774539136 model_lib_v2.py:799] Finished eval step 8100
    INFO:tensorflow:Finished eval step 8200
    I1009 17:38:24.433124 140592774539136 model_lib_v2.py:799] Finished eval step 8200
    INFO:tensorflow:Finished eval step 8300
    I1009 17:38:31.536814 140592774539136 model_lib_v2.py:799] Finished eval step 8300
    INFO:tensorflow:Finished eval step 8400
    I1009 17:38:38.803203 140592774539136 model_lib_v2.py:799] Finished eval step 8400
    INFO:tensorflow:Finished eval step 8500
    I1009 17:38:45.935532 140592774539136 model_lib_v2.py:799] Finished eval step 8500
    INFO:tensorflow:Finished eval step 8600
    I1009 17:38:53.100721 140592774539136 model_lib_v2.py:799] Finished eval step 8600
    INFO:tensorflow:Finished eval step 8700
    I1009 17:39:00.299176 140592774539136 model_lib_v2.py:799] Finished eval step 8700
    INFO:tensorflow:Finished eval step 8800
    I1009 17:39:07.494309 140592774539136 model_lib_v2.py:799] Finished eval step 8800
    INFO:tensorflow:Finished eval step 8900
    I1009 17:39:14.685568 140592774539136 model_lib_v2.py:799] Finished eval step 8900
    INFO:tensorflow:Finished eval step 9000
    I1009 17:39:21.933660 140592774539136 model_lib_v2.py:799] Finished eval step 9000
    INFO:tensorflow:Finished eval step 9100
    I1009 17:39:30.208340 140592774539136 model_lib_v2.py:799] Finished eval step 9100
    INFO:tensorflow:Finished eval step 9200
    I1009 17:39:37.374509 140592774539136 model_lib_v2.py:799] Finished eval step 9200
    INFO:tensorflow:Finished eval step 9300
    I1009 17:39:44.482306 140592774539136 model_lib_v2.py:799] Finished eval step 9300
    INFO:tensorflow:Finished eval step 9400
    I1009 17:39:51.696965 140592774539136 model_lib_v2.py:799] Finished eval step 9400
    INFO:tensorflow:Finished eval step 9500
    I1009 17:39:58.887935 140592774539136 model_lib_v2.py:799] Finished eval step 9500
    INFO:tensorflow:Finished eval step 9600
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
    I1009 17:47:55.744640 140592774539136 model_lib_v2.py:853] Eval metrics at step 150000
    INFO:tensorflow:	+ DetectionBoxes_Precision/mAP: 0.142052
    I1009 17:47:55.767384 140592774539136 model_lib_v2.py:856] 	+ DetectionBoxes_Precision/mAP: 0.142052
    INFO:tensorflow:	+ DetectionBoxes_Precision/mAP@.50IOU: 0.288094
    I1009 17:47:55.769040 140592774539136 model_lib_v2.py:856] 	+ DetectionBoxes_Precision/mAP@.50IOU: 0.288094
    INFO:tensorflow:	+ DetectionBoxes_Precision/mAP@.75IOU: 0.120198
    I1009 17:47:55.770462 140592774539136 model_lib_v2.py:856] 	+ DetectionBoxes_Precision/mAP@.75IOU: 0.120198
    INFO:tensorflow:	+ DetectionBoxes_Precision/mAP (small): 0.039086
    I1009 17:47:55.771786 140592774539136 model_lib_v2.py:856] 	+ DetectionBoxes_Precision/mAP (small): 0.039086
    INFO:tensorflow:	+ DetectionBoxes_Precision/mAP (medium): 0.219396
    I1009 17:47:55.773159 140592774539136 model_lib_v2.py:856] 	+ DetectionBoxes_Precision/mAP (medium): 0.219396
    INFO:tensorflow:	+ DetectionBoxes_Precision/mAP (large): 0.396311
    I1009 17:47:55.774445 140592774539136 model_lib_v2.py:856] 	+ DetectionBoxes_Precision/mAP (large): 0.396311
    INFO:tensorflow:	+ DetectionBoxes_Recall/AR@1: 0.164296
    I1009 17:47:55.775747 140592774539136 model_lib_v2.py:856] 	+ DetectionBoxes_Recall/AR@1: 0.164296
    INFO:tensorflow:	+ DetectionBoxes_Recall/AR@10: 0.283512
    I1009 17:47:55.777097 140592774539136 model_lib_v2.py:856] 	+ DetectionBoxes_Recall/AR@10: 0.283512
    INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100: 0.297494
    I1009 17:47:55.778366 140592774539136 model_lib_v2.py:856] 	+ DetectionBoxes_Recall/AR@100: 0.297494
    INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100 (small): 0.113009
    I1009 17:47:55.779751 140592774539136 model_lib_v2.py:856] 	+ DetectionBoxes_Recall/AR@100 (small): 0.113009
    INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100 (medium): 0.384964
    I1009 17:47:55.781040 140592774539136 model_lib_v2.py:856] 	+ DetectionBoxes_Recall/AR@100 (medium): 0.384964
    INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100 (large): 0.570223
    I1009 17:47:55.782587 140592774539136 model_lib_v2.py:856] 	+ DetectionBoxes_Recall/AR@100 (large): 0.570223
    INFO:tensorflow:	+ Loss/RPNLoss/localization_loss: 0.239309
    I1009 17:47:55.784541 140592774539136 model_lib_v2.py:856] 	+ Loss/RPNLoss/localization_loss: 0.239309
    INFO:tensorflow:	+ Loss/RPNLoss/objectness_loss: 0.028339
    I1009 17:47:55.785884 140592774539136 model_lib_v2.py:856] 	+ Loss/RPNLoss/objectness_loss: 0.028339
    INFO:tensorflow:	+ Loss/BoxClassifierLoss/localization_loss: 0.209647
    I1009 17:47:55.787105 140592774539136 model_lib_v2.py:856] 	+ Loss/BoxClassifierLoss/localization_loss: 0.209647
    INFO:tensorflow:	+ Loss/BoxClassifierLoss/classification_loss: 0.164453
    I1009 17:47:55.788279 140592774539136 model_lib_v2.py:856] 	+ Loss/BoxClassifierLoss/classification_loss: 0.164453
    INFO:tensorflow:	+ Loss/regularization_loss: 0.000000
    I1009 17:47:55.789300 140592774539136 model_lib_v2.py:856] 	+ Loss/regularization_loss: 0.000000
    INFO:tensorflow:	+ Loss/total_loss: 0.641706
    I1009 17:47:55.790439 140592774539136 model_lib_v2.py:856] 	+ Loss/total_loss: 0.641706
    INFO:tensorflow:Waiting for new checkpoint at ./myoutputmodel
    I1009 17:47:57.119554 140592774539136 checkpoint_utils.py:125] Waiting for new checkpoint at ./myoutputmodel
    Traceback (most recent call last):
      File "/usr/local/lib/python3.6/dist-packages/absl/app.py", line 300, in run
        _run_main(main, args)
      File "/usr/local/lib/python3.6/dist-packages/absl/app.py", line 251, in _run_main
        sys.exit(main(argv))
      File "models/research/object_detection/model_main_tf2.py", line 88, in main
        wait_interval=300, timeout=FLAGS.eval_timeout)
      File "/usr/local/lib/python3.6/dist-packages/object_detection/model_lib_v2.py", line 966, in eval_continuously
        checkpoint_dir, timeout=timeout, min_interval_secs=wait_interval):
      File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/training/checkpoint_utils.py", line 184, in checkpoints_iterator
        checkpoint_dir, checkpoint_path, timeout=timeout)
      File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/training/checkpoint_utils.py", line 132, in wait_for_new_checkpoint
        time.sleep(seconds_to_sleep)
    KeyboardInterrupt
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "models/research/object_detection/model_main_tf2.py", line 113, in <module>
        tf.compat.v1.app.run()
      File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/platform/app.py", line 40, in run
        _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
      File "/usr/local/lib/python3.6/dist-packages/absl/app.py", line 313, in run
        if FLAGS.pdb_post_mortem and sys.stdout.isatty():
      File "/usr/local/lib/python3.6/dist-packages/absl/flags/_flagvalues.py", line 478, in __getattr__
        fl = self._flags()
    KeyboardInterrupt


# Evaluation results at 150k steps



```
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




```python
!cp drive/My\ Drive/train.record .
!cp drive/My\ Drive/test.record .
!cp drive/My\ Drive/pipeline_file.config .
!cp drive/My\ Drive/label_map.pbtxt .
!cp drive/My\ Drive/myoutputmodel_150k.zip .
!unzip myoutputmodel_150k.zip
!rm myoutputmodel_150k.zip
```

    Archive:  myoutputmodel_150k.zip
       creating: myoutputmodel/
      inflating: myoutputmodel/ckpt-145.data-00000-of-00001  
      inflating: myoutputmodel/ckpt-149.data-00000-of-00001  
      inflating: myoutputmodel/ckpt-150.index  
      inflating: myoutputmodel/ckpt-147.index  
      inflating: myoutputmodel/ckpt-149.index  
      inflating: myoutputmodel/ckpt-146.data-00000-of-00001  
       creating: myoutputmodel/train/
      inflating: myoutputmodel/train/events.out.tfevents.1602222489.572f21843685.3333.1504.v2  
      inflating: myoutputmodel/ckpt-146.index  
      inflating: myoutputmodel/ckpt-151.data-00000-of-00001  
      inflating: myoutputmodel/ckpt-151.index  
      inflating: myoutputmodel/ckpt-148.index  
      inflating: myoutputmodel/ckpt-147.data-00000-of-00001  
      inflating: myoutputmodel/ckpt-150.data-00000-of-00001  
      inflating: myoutputmodel/checkpoint  
      inflating: myoutputmodel/ckpt-148.data-00000-of-00001  
      inflating: myoutputmodel/ckpt-145.index  


# Perform additional training for another 100k steps


```python
!python models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={'./pipeline_file.config'} \
    --model_dir={'./myoutputmodel'} \
    --alsologtostderr
```

    2020-10-10 21:26:04.017587: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-10 21:26:10.697446: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
    2020-10-10 21:26:10.761858: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-10 21:26:10.762572: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
    coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.75GiB deviceMemoryBandwidth: 836.37GiB/s
    2020-10-10 21:26:10.762625: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-10 21:26:11.055106: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2020-10-10 21:26:11.207024: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
    2020-10-10 21:26:11.235600: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
    2020-10-10 21:26:11.552672: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
    2020-10-10 21:26:11.595746: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
    2020-10-10 21:26:12.162916: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    2020-10-10 21:26:12.163150: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-10 21:26:12.163895: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-10 21:26:12.164548: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
    2020-10-10 21:26:12.172272: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX512F
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2020-10-10 21:26:12.212284: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2000170000 Hz
    2020-10-10 21:26:12.212922: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x2e07640 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-10-10 21:26:12.212966: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-10-10 21:26:12.390318: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-10 21:26:12.391219: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x2e07800 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2020-10-10 21:26:12.391256: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
    2020-10-10 21:26:12.397449: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-10 21:26:12.398094: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
    coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.75GiB deviceMemoryBandwidth: 836.37GiB/s
    2020-10-10 21:26:12.398136: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-10 21:26:12.398174: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2020-10-10 21:26:12.398200: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
    2020-10-10 21:26:12.398214: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
    2020-10-10 21:26:12.398226: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
    2020-10-10 21:26:12.398238: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
    2020-10-10 21:26:12.398250: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    2020-10-10 21:26:12.398357: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-10 21:26:12.399087: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-10 21:26:12.399682: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
    2020-10-10 21:26:12.404160: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-10 21:26:16.306522: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-10-10 21:26:16.306592: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0 
    2020-10-10 21:26:16.306606: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N 
    2020-10-10 21:26:16.313136: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-10 21:26:16.313969: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-10 21:26:16.314577: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
    2020-10-10 21:26:16.314626: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14756 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0)
    INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0',)
    I1010 21:26:16.322095 139761090996096 mirrored_strategy.py:341] Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0',)
    INFO:tensorflow:Maybe overwriting train_steps: None
    I1010 21:26:16.329381 139761090996096 config_util.py:552] Maybe overwriting train_steps: None
    INFO:tensorflow:Maybe overwriting use_bfloat16: False
    I1010 21:26:16.329578 139761090996096 config_util.py:552] Maybe overwriting use_bfloat16: False
    INFO:tensorflow:Reading unweighted datasets: ['train.record']
    I1010 21:26:16.448585 139761090996096 dataset_builder.py:148] Reading unweighted datasets: ['train.record']
    INFO:tensorflow:Reading record datasets for input file: ['train.record']
    I1010 21:26:16.450458 139761090996096 dataset_builder.py:77] Reading record datasets for input file: ['train.record']
    INFO:tensorflow:Number of filenames to read: 1
    I1010 21:26:16.450606 139761090996096 dataset_builder.py:78] Number of filenames to read: 1
    WARNING:tensorflow:num_readers has been reduced to 1 to match input file shards.
    W1010 21:26:16.450676 139761090996096 dataset_builder.py:86] num_readers has been reduced to 1 to match input file shards.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:103: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.experimental_deterministic`.
    W1010 21:26:16.469820 139761090996096 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:103: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.experimental_deterministic`.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:222: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.map()
    W1010 21:26:16.542411 139761090996096 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:222: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.map()
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
    W1010 21:26:23.773789 139761090996096 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/inputs.py:262: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    W1010 21:26:26.694558 139761090996096 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/inputs.py:262: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib_v2.py:355: set_learning_phase (from tensorflow.python.keras.backend) is deprecated and will be removed after 2020-10-11.
    Instructions for updating:
    Simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
    W1010 21:26:34.632246 139757402904320 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib_v2.py:355: set_learning_phase (from tensorflow.python.keras.backend) is deprecated and will be removed after 2020-10-11.
    Instructions for updating:
    Simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
    INFO:tensorflow:depth of additional conv before box predictor: 0
    I1010 21:26:39.545670 139757402904320 convolutional_keras_box_predictor.py:154] depth of additional conv before box predictor: 0
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/model_util.py:57: Tensor.experimental_ref (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use ref() instead.
    W1010 21:26:48.103679 139757402904320 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/model_util.py:57: Tensor.experimental_ref (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use ref() instead.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    W1010 21:26:53.808073 139757402904320 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    2020-10-10 21:27:09.017343: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2020-10-10 21:27:10.516403: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._groundtruth_lists
    W1010 21:27:17.639780 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._groundtruth_lists
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv
    W1010 21:27:17.640149 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor
    W1010 21:27:17.640224 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._maxpool_layer
    W1010 21:27:17.640282 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._maxpool_layer
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor
    W1010 21:27:17.640333 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._batched_prediction_tensor_names
    W1010 21:27:17.640382 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._batched_prediction_tensor_names
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model.endpoints
    W1010 21:27:17.640430 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model.endpoints
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0
    W1010 21:27:17.640477 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-1
    W1010 21:27:17.640524 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-1
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-2
    W1010 21:27:17.640570 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-2
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads
    W1010 21:27:17.640615 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._sorted_head_names
    W1010 21:27:17.640661 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._sorted_head_names
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._shared_nets
    W1010 21:27:17.640718 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._shared_nets
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head
    W1010 21:27:17.640766 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head
    W1010 21:27:17.640811 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._third_stage_heads
    W1010 21:27:17.640857 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._third_stage_heads
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0._inbound_nodes
    W1010 21:27:17.640916 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0._inbound_nodes
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0.kernel
    W1010 21:27:17.640965 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0.kernel
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0.bias
    W1010 21:27:17.641013 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer_with_weights-0.bias
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-1._inbound_nodes
    W1010 21:27:17.641059 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-1._inbound_nodes
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-2._inbound_nodes
    W1010 21:27:17.641105 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor_first_conv.layer-2._inbound_nodes
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings
    W1010 21:27:17.641150 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background
    W1010 21:27:17.641196 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._shared_nets.0
    W1010 21:27:17.641248 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._shared_nets.0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers
    W1010 21:27:17.641294 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers
    W1010 21:27:17.641340 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0
    W1010 21:27:17.641433 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0
    W1010 21:27:17.641481 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.0
    W1010 21:27:17.641529 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.1
    W1010 21:27:17.641574 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.1
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.2
    W1010 21:27:17.641620 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.2
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.0
    W1010 21:27:17.641667 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.1
    W1010 21:27:17.641723 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.1
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.2
    W1010 21:27:17.641771 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.2
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers
    W1010 21:27:17.641817 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers
    W1010 21:27:17.641864 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.1.kernel
    W1010 21:27:17.641919 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.1.kernel
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.1.bias
    W1010 21:27:17.641966 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._box_prediction_head._box_encoder_layers.1.bias
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.1.kernel
    W1010 21:27:17.642011 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.1.kernel
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.1.bias
    W1010 21:27:17.642058 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._mask_rcnn_box_predictor._class_prediction_head._class_predictor_layers.1.bias
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers.0
    W1010 21:27:17.642108 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers.0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers.0
    W1010 21:27:17.642154 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers.0
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers.0.kernel
    W1010 21:27:17.642201 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers.0.kernel
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers.0.bias
    W1010 21:27:17.642248 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.box_encodings.0._box_encoder_layers.0.bias
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers.0.kernel
    W1010 21:27:17.642293 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers.0.kernel
    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers.0.bias
    W1010 21:27:17.642341 139761090996096 util.py:150] Unresolved object in checkpoint: (root).model._first_stage_box_predictor._prediction_heads.class_predictions_with_background.0._class_predictor_layers.0.bias
    WARNING:tensorflow:A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.
    W1010 21:27:17.642390 139761090996096 util.py:158] A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/deprecation.py:574: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Use fn_output_signature instead
    W1010 21:27:29.796357 139757444867840 deprecation.py:506] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/deprecation.py:574: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Use fn_output_signature instead
    INFO:tensorflow:Step 150100 per-step time 0.246s loss=0.745
    I1010 21:28:13.789007 139761090996096 model_lib_v2.py:652] Step 150100 per-step time 0.246s loss=0.745
    INFO:tensorflow:Step 150200 per-step time 0.249s loss=1.228
    I1010 21:28:38.804928 139761090996096 model_lib_v2.py:652] Step 150200 per-step time 0.249s loss=1.228
    INFO:tensorflow:Step 150300 per-step time 0.248s loss=0.564
    I1010 21:29:04.066321 139761090996096 model_lib_v2.py:652] Step 150300 per-step time 0.248s loss=0.564
    INFO:tensorflow:Step 150400 per-step time 0.257s loss=0.602
    I1010 21:29:29.192486 139761090996096 model_lib_v2.py:652] Step 150400 per-step time 0.257s loss=0.602
    INFO:tensorflow:Step 150500 per-step time 0.257s loss=0.645
    I1010 21:29:54.226896 139761090996096 model_lib_v2.py:652] Step 150500 per-step time 0.257s loss=0.645
    INFO:tensorflow:Step 150600 per-step time 0.252s loss=0.654
    I1010 21:30:19.334540 139761090996096 model_lib_v2.py:652] Step 150600 per-step time 0.252s loss=0.654
    INFO:tensorflow:Step 150700 per-step time 0.239s loss=0.601
    I1010 21:30:44.498881 139761090996096 model_lib_v2.py:652] Step 150700 per-step time 0.239s loss=0.601
    INFO:tensorflow:Step 150800 per-step time 0.252s loss=0.820
    I1010 21:31:09.567090 139761090996096 model_lib_v2.py:652] Step 150800 per-step time 0.252s loss=0.820
    INFO:tensorflow:Step 150900 per-step time 0.239s loss=0.360
    I1010 21:31:34.958573 139761090996096 model_lib_v2.py:652] Step 150900 per-step time 0.239s loss=0.360
    INFO:tensorflow:Step 151000 per-step time 0.239s loss=0.723
    I1010 21:31:59.957572 139761090996096 model_lib_v2.py:652] Step 151000 per-step time 0.239s loss=0.723
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1010 21:32:00.064729 139761090996096 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1010 21:32:00.066181 139761090996096 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1010 21:32:00.069286 139761090996096 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1010 21:32:00.070246 139761090996096 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1010 21:32:00.072214 139761090996096 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1010 21:32:00.073145 139761090996096 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1010 21:32:00.075403 139761090996096 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1010 21:32:00.076290 139761090996096 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1010 21:32:00.077938 139761090996096 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    I1010 21:32:00.079131 139761090996096 cross_device_ops.py:443] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Step 151100 per-step time 0.260s loss=0.561
    I1010 21:32:25.947839 139761090996096 model_lib_v2.py:652] Step 151100 per-step time 0.260s loss=0.561
    INFO:tensorflow:Step 151200 per-step time 0.258s loss=0.557
    I1010 21:32:51.077225 139761090996096 model_lib_v2.py:652] Step 151200 per-step time 0.258s loss=0.557
    INFO:tensorflow:Step 151300 per-step time 0.255s loss=0.918
    I1010 21:33:16.327776 139761090996096 model_lib_v2.py:652] Step 151300 per-step time 0.255s loss=0.918
    INFO:tensorflow:Step 151400 per-step time 0.251s loss=0.617
    I1010 21:33:41.591910 139761090996096 model_lib_v2.py:652] Step 151400 per-step time 0.251s loss=0.617
    INFO:tensorflow:Step 151500 per-step time 0.254s loss=0.615
    I1010 21:34:06.730103 139761090996096 model_lib_v2.py:652] Step 151500 per-step time 0.254s loss=0.615
    INFO:tensorflow:Step 151600 per-step time 0.258s loss=0.719
    I1010 21:34:31.929005 139761090996096 model_lib_v2.py:652] Step 151600 per-step time 0.258s loss=0.719
    INFO:tensorflow:Step 151700 per-step time 0.259s loss=0.725
    I1010 21:34:57.000653 139761090996096 model_lib_v2.py:652] Step 151700 per-step time 0.259s loss=0.725
    INFO:tensorflow:Step 151800 per-step time 0.248s loss=0.372
    I1010 21:35:22.209093 139761090996096 model_lib_v2.py:652] Step 151800 per-step time 0.248s loss=0.372
    INFO:tensorflow:Step 151900 per-step time 0.256s loss=0.692
    I1010 21:35:47.094036 139761090996096 model_lib_v2.py:652] Step 151900 per-step time 0.256s loss=0.692
    INFO:tensorflow:Step 152000 per-step time 0.244s loss=0.585
    I1010 21:36:12.256793 139761090996096 model_lib_v2.py:652] Step 152000 per-step time 0.244s loss=0.585
    INFO:tensorflow:Step 152100 per-step time 0.251s loss=0.607
    I1010 21:36:38.421738 139761090996096 model_lib_v2.py:652] Step 152100 per-step time 0.251s loss=0.607
    INFO:tensorflow:Step 152200 per-step time 0.240s loss=0.663
    I1010 21:37:03.353074 139761090996096 model_lib_v2.py:652] Step 152200 per-step time 0.240s loss=0.663
    INFO:tensorflow:Step 152300 per-step time 0.247s loss=0.624
    I1010 21:37:28.334875 139761090996096 model_lib_v2.py:652] Step 152300 per-step time 0.247s loss=0.624
    INFO:tensorflow:Step 152400 per-step time 0.243s loss=0.518
    I1010 21:37:53.512344 139761090996096 model_lib_v2.py:652] Step 152400 per-step time 0.243s loss=0.518
    INFO:tensorflow:Step 152500 per-step time 0.252s loss=0.583
    I1010 21:38:18.703579 139761090996096 model_lib_v2.py:652] Step 152500 per-step time 0.252s loss=0.583
    INFO:tensorflow:Step 152600 per-step time 0.248s loss=0.716
    I1010 21:38:43.742069 139761090996096 model_lib_v2.py:652] Step 152600 per-step time 0.248s loss=0.716
    INFO:tensorflow:Step 152700 per-step time 0.260s loss=0.818
    I1010 21:39:08.908370 139761090996096 model_lib_v2.py:652] Step 152700 per-step time 0.260s loss=0.818
    INFO:tensorflow:Step 152800 per-step time 0.250s loss=0.625
    I1010 21:39:33.959138 139761090996096 model_lib_v2.py:652] Step 152800 per-step time 0.250s loss=0.625
    INFO:tensorflow:Step 152900 per-step time 0.251s loss=0.581
    I1010 21:39:59.081298 139761090996096 model_lib_v2.py:652] Step 152900 per-step time 0.251s loss=0.581
    INFO:tensorflow:Step 153000 per-step time 0.237s loss=0.882
    I1010 21:40:24.122001 139761090996096 model_lib_v2.py:652] Step 153000 per-step time 0.237s loss=0.882
    INFO:tensorflow:Step 153100 per-step time 0.245s loss=0.769
    I1010 21:40:50.095654 139761090996096 model_lib_v2.py:652] Step 153100 per-step time 0.245s loss=0.769
    INFO:tensorflow:Step 153200 per-step time 0.245s loss=1.149
    I1010 21:41:15.280782 139761090996096 model_lib_v2.py:652] Step 153200 per-step time 0.245s loss=1.149
    INFO:tensorflow:Step 153300 per-step time 0.255s loss=0.561
    I1010 21:41:40.300082 139761090996096 model_lib_v2.py:652] Step 153300 per-step time 0.255s loss=0.561
    INFO:tensorflow:Step 153400 per-step time 0.257s loss=0.458
    I1010 21:42:05.630974 139761090996096 model_lib_v2.py:652] Step 153400 per-step time 0.257s loss=0.458
    INFO:tensorflow:Step 153500 per-step time 0.252s loss=0.655
    I1010 21:42:30.725190 139761090996096 model_lib_v2.py:652] Step 153500 per-step time 0.252s loss=0.655
    INFO:tensorflow:Step 153600 per-step time 0.260s loss=0.673
    I1010 21:42:55.882045 139761090996096 model_lib_v2.py:652] Step 153600 per-step time 0.260s loss=0.673
    INFO:tensorflow:Step 153700 per-step time 0.255s loss=0.594
    I1010 21:43:20.917535 139761090996096 model_lib_v2.py:652] Step 153700 per-step time 0.255s loss=0.594
    INFO:tensorflow:Step 153800 per-step time 0.263s loss=0.619
    I1010 21:43:45.904246 139761090996096 model_lib_v2.py:652] Step 153800 per-step time 0.263s loss=0.619
    INFO:tensorflow:Step 153900 per-step time 0.236s loss=0.604
    I1010 21:44:11.004496 139761090996096 model_lib_v2.py:652] Step 153900 per-step time 0.236s loss=0.604
    INFO:tensorflow:Step 154000 per-step time 0.242s loss=0.708
    I1010 21:44:35.963920 139761090996096 model_lib_v2.py:652] Step 154000 per-step time 0.242s loss=0.708
    INFO:tensorflow:Step 154100 per-step time 0.251s loss=0.644
    I1010 21:45:01.938354 139761090996096 model_lib_v2.py:652] Step 154100 per-step time 0.251s loss=0.644
    INFO:tensorflow:Step 154200 per-step time 0.255s loss=0.538
    I1010 21:45:27.125007 139761090996096 model_lib_v2.py:652] Step 154200 per-step time 0.255s loss=0.538
    INFO:tensorflow:Step 154300 per-step time 0.244s loss=0.756
    I1010 21:45:52.226396 139761090996096 model_lib_v2.py:652] Step 154300 per-step time 0.244s loss=0.756
    INFO:tensorflow:Step 154400 per-step time 0.252s loss=0.701
    I1010 21:46:17.325939 139761090996096 model_lib_v2.py:652] Step 154400 per-step time 0.252s loss=0.701
    INFO:tensorflow:Step 154500 per-step time 0.259s loss=1.079
    I1010 21:46:42.388348 139761090996096 model_lib_v2.py:652] Step 154500 per-step time 0.259s loss=1.079
    INFO:tensorflow:Step 154600 per-step time 0.249s loss=0.753
    I1010 21:47:07.676724 139761090996096 model_lib_v2.py:652] Step 154600 per-step time 0.249s loss=0.753
    INFO:tensorflow:Step 154700 per-step time 0.264s loss=0.735
    I1010 21:47:32.922429 139761090996096 model_lib_v2.py:652] Step 154700 per-step time 0.264s loss=0.735
    INFO:tensorflow:Step 154800 per-step time 0.236s loss=0.605
    I1010 21:47:58.095881 139761090996096 model_lib_v2.py:652] Step 154800 per-step time 0.236s loss=0.605
    INFO:tensorflow:Step 154900 per-step time 0.246s loss=0.839
    I1010 21:48:23.033812 139761090996096 model_lib_v2.py:652] Step 154900 per-step time 0.246s loss=0.839
    INFO:tensorflow:Step 155000 per-step time 0.264s loss=0.626
    I1010 21:48:48.205284 139761090996096 model_lib_v2.py:652] Step 155000 per-step time 0.264s loss=0.626
    INFO:tensorflow:Step 155100 per-step time 0.256s loss=0.735
    I1010 21:49:14.089341 139761090996096 model_lib_v2.py:652] Step 155100 per-step time 0.256s loss=0.735
    INFO:tensorflow:Step 155200 per-step time 0.254s loss=0.853
    I1010 21:49:39.169281 139761090996096 model_lib_v2.py:652] Step 155200 per-step time 0.254s loss=0.853
    INFO:tensorflow:Step 155300 per-step time 0.237s loss=0.507
    I1010 21:50:04.266019 139761090996096 model_lib_v2.py:652] Step 155300 per-step time 0.237s loss=0.507
    INFO:tensorflow:Step 155400 per-step time 0.255s loss=0.820
    I1010 21:50:29.237559 139761090996096 model_lib_v2.py:652] Step 155400 per-step time 0.255s loss=0.820
    INFO:tensorflow:Step 155500 per-step time 0.245s loss=0.567
    I1010 21:50:54.330689 139761090996096 model_lib_v2.py:652] Step 155500 per-step time 0.245s loss=0.567
    INFO:tensorflow:Step 155600 per-step time 0.250s loss=0.519
    I1010 21:51:19.477689 139761090996096 model_lib_v2.py:652] Step 155600 per-step time 0.250s loss=0.519
    INFO:tensorflow:Step 155700 per-step time 0.261s loss=0.819
    I1010 21:51:44.718604 139761090996096 model_lib_v2.py:652] Step 155700 per-step time 0.261s loss=0.819
    INFO:tensorflow:Step 155800 per-step time 0.246s loss=0.528
    I1010 21:52:09.965995 139761090996096 model_lib_v2.py:652] Step 155800 per-step time 0.246s loss=0.528
    INFO:tensorflow:Step 155900 per-step time 0.242s loss=0.602
    I1010 21:52:35.003942 139761090996096 model_lib_v2.py:652] Step 155900 per-step time 0.242s loss=0.602
    INFO:tensorflow:Step 156000 per-step time 0.249s loss=0.769
    I1010 21:53:00.066023 139761090996096 model_lib_v2.py:652] Step 156000 per-step time 0.249s loss=0.769
    INFO:tensorflow:Step 156100 per-step time 0.252s loss=0.620
    I1010 21:53:26.681153 139761090996096 model_lib_v2.py:652] Step 156100 per-step time 0.252s loss=0.620
    INFO:tensorflow:Step 156200 per-step time 0.251s loss=0.642
    I1010 21:53:51.631433 139761090996096 model_lib_v2.py:652] Step 156200 per-step time 0.251s loss=0.642
    INFO:tensorflow:Step 156300 per-step time 0.258s loss=0.499
    I1010 21:54:16.764230 139761090996096 model_lib_v2.py:652] Step 156300 per-step time 0.258s loss=0.499
    INFO:tensorflow:Step 156400 per-step time 0.262s loss=0.445
    I1010 21:54:41.735335 139761090996096 model_lib_v2.py:652] Step 156400 per-step time 0.262s loss=0.445
    INFO:tensorflow:Step 156500 per-step time 0.246s loss=0.857
    I1010 21:55:06.831438 139761090996096 model_lib_v2.py:652] Step 156500 per-step time 0.246s loss=0.857
    INFO:tensorflow:Step 156600 per-step time 0.257s loss=0.575
    I1010 21:55:31.843477 139761090996096 model_lib_v2.py:652] Step 156600 per-step time 0.257s loss=0.575
    INFO:tensorflow:Step 156700 per-step time 0.255s loss=0.722
    I1010 21:55:56.806798 139761090996096 model_lib_v2.py:652] Step 156700 per-step time 0.255s loss=0.722
    INFO:tensorflow:Step 156800 per-step time 0.241s loss=0.358
    I1010 21:56:21.909179 139761090996096 model_lib_v2.py:652] Step 156800 per-step time 0.241s loss=0.358
    INFO:tensorflow:Step 156900 per-step time 0.254s loss=1.017
    I1010 21:56:46.906501 139761090996096 model_lib_v2.py:652] Step 156900 per-step time 0.254s loss=1.017
    INFO:tensorflow:Step 157000 per-step time 0.249s loss=0.548
    I1010 21:57:11.837662 139761090996096 model_lib_v2.py:652] Step 157000 per-step time 0.249s loss=0.548
    INFO:tensorflow:Step 157100 per-step time 0.246s loss=0.751
    I1010 21:57:37.806652 139761090996096 model_lib_v2.py:652] Step 157100 per-step time 0.246s loss=0.751
    INFO:tensorflow:Step 157200 per-step time 0.251s loss=0.481
    I1010 21:58:02.764493 139761090996096 model_lib_v2.py:652] Step 157200 per-step time 0.251s loss=0.481
    INFO:tensorflow:Step 157300 per-step time 0.246s loss=0.896
    I1010 21:58:27.727429 139761090996096 model_lib_v2.py:652] Step 157300 per-step time 0.246s loss=0.896
    INFO:tensorflow:Step 157400 per-step time 0.246s loss=0.460
    I1010 21:58:52.641910 139761090996096 model_lib_v2.py:652] Step 157400 per-step time 0.246s loss=0.460
    INFO:tensorflow:Step 157500 per-step time 0.248s loss=0.526
    I1010 21:59:17.766124 139761090996096 model_lib_v2.py:652] Step 157500 per-step time 0.248s loss=0.526
    INFO:tensorflow:Step 157600 per-step time 0.247s loss=0.864
    I1010 21:59:42.803037 139761090996096 model_lib_v2.py:652] Step 157600 per-step time 0.247s loss=0.864
    INFO:tensorflow:Step 157700 per-step time 0.257s loss=0.630
    I1010 22:00:07.695443 139761090996096 model_lib_v2.py:652] Step 157700 per-step time 0.257s loss=0.630
    INFO:tensorflow:Step 157800 per-step time 0.263s loss=0.633
    I1010 22:00:32.570311 139761090996096 model_lib_v2.py:652] Step 157800 per-step time 0.263s loss=0.633
    INFO:tensorflow:Step 157900 per-step time 0.252s loss=0.825
    I1010 22:00:57.458535 139761090996096 model_lib_v2.py:652] Step 157900 per-step time 0.252s loss=0.825
    INFO:tensorflow:Step 158000 per-step time 0.257s loss=0.886
    I1010 22:01:22.577661 139761090996096 model_lib_v2.py:652] Step 158000 per-step time 0.257s loss=0.886
    INFO:tensorflow:Step 158100 per-step time 0.258s loss=0.487
    I1010 22:01:48.483917 139761090996096 model_lib_v2.py:652] Step 158100 per-step time 0.258s loss=0.487
    INFO:tensorflow:Step 158200 per-step time 0.239s loss=0.536
    I1010 22:02:13.509700 139761090996096 model_lib_v2.py:652] Step 158200 per-step time 0.239s loss=0.536
    INFO:tensorflow:Step 158300 per-step time 0.256s loss=0.617
    I1010 22:02:38.624645 139761090996096 model_lib_v2.py:652] Step 158300 per-step time 0.256s loss=0.617
    INFO:tensorflow:Step 158400 per-step time 0.249s loss=0.728
    I1010 22:03:03.783888 139761090996096 model_lib_v2.py:652] Step 158400 per-step time 0.249s loss=0.728
    INFO:tensorflow:Step 158500 per-step time 0.248s loss=1.057
    I1010 22:03:28.744230 139761090996096 model_lib_v2.py:652] Step 158500 per-step time 0.248s loss=1.057
    INFO:tensorflow:Step 158600 per-step time 0.247s loss=0.639
    I1010 22:03:53.787079 139761090996096 model_lib_v2.py:652] Step 158600 per-step time 0.247s loss=0.639
    INFO:tensorflow:Step 158700 per-step time 0.258s loss=0.582
    I1010 22:04:18.673856 139761090996096 model_lib_v2.py:652] Step 158700 per-step time 0.258s loss=0.582
    INFO:tensorflow:Step 158800 per-step time 0.254s loss=0.846
    I1010 22:04:43.721978 139761090996096 model_lib_v2.py:652] Step 158800 per-step time 0.254s loss=0.846
    INFO:tensorflow:Step 158900 per-step time 0.256s loss=0.912
    I1010 22:05:08.818472 139761090996096 model_lib_v2.py:652] Step 158900 per-step time 0.256s loss=0.912
    INFO:tensorflow:Step 159000 per-step time 0.256s loss=0.805
    I1010 22:05:33.777183 139761090996096 model_lib_v2.py:652] Step 159000 per-step time 0.256s loss=0.805
    INFO:tensorflow:Step 159100 per-step time 0.258s loss=0.877
    I1010 22:05:59.596831 139761090996096 model_lib_v2.py:652] Step 159100 per-step time 0.258s loss=0.877
    INFO:tensorflow:Step 159200 per-step time 0.254s loss=0.393
    I1010 22:06:24.462123 139761090996096 model_lib_v2.py:652] Step 159200 per-step time 0.254s loss=0.393
    INFO:tensorflow:Step 159300 per-step time 0.254s loss=0.523
    I1010 22:06:49.647760 139761090996096 model_lib_v2.py:652] Step 159300 per-step time 0.254s loss=0.523
    INFO:tensorflow:Step 159400 per-step time 0.249s loss=0.775
    I1010 22:07:14.653558 139761090996096 model_lib_v2.py:652] Step 159400 per-step time 0.249s loss=0.775
    INFO:tensorflow:Step 159500 per-step time 0.255s loss=0.833
    I1010 22:07:39.741264 139761090996096 model_lib_v2.py:652] Step 159500 per-step time 0.255s loss=0.833
    INFO:tensorflow:Step 159600 per-step time 0.239s loss=0.715
    I1010 22:08:04.768772 139761090996096 model_lib_v2.py:652] Step 159600 per-step time 0.239s loss=0.715
    INFO:tensorflow:Step 159700 per-step time 0.241s loss=0.608
    I1010 22:08:29.726623 139761090996096 model_lib_v2.py:652] Step 159700 per-step time 0.241s loss=0.608
    INFO:tensorflow:Step 159800 per-step time 0.248s loss=0.800
    I1010 22:08:54.749700 139761090996096 model_lib_v2.py:652] Step 159800 per-step time 0.248s loss=0.800
    INFO:tensorflow:Step 159900 per-step time 0.245s loss=0.683
    I1010 22:09:19.675496 139761090996096 model_lib_v2.py:652] Step 159900 per-step time 0.245s loss=0.683
    INFO:tensorflow:Step 160000 per-step time 0.255s loss=0.946
    I1010 22:09:44.588135 139761090996096 model_lib_v2.py:652] Step 160000 per-step time 0.255s loss=0.946
    INFO:tensorflow:Step 160100 per-step time 0.256s loss=0.401
    I1010 22:10:10.415570 139761090996096 model_lib_v2.py:652] Step 160100 per-step time 0.256s loss=0.401
    INFO:tensorflow:Step 160200 per-step time 0.255s loss=0.669
    I1010 22:10:35.384956 139761090996096 model_lib_v2.py:652] Step 160200 per-step time 0.255s loss=0.669
    INFO:tensorflow:Step 160300 per-step time 0.243s loss=0.686
    I1010 22:11:00.299406 139761090996096 model_lib_v2.py:652] Step 160300 per-step time 0.243s loss=0.686
    INFO:tensorflow:Step 160400 per-step time 0.248s loss=1.378
    I1010 22:11:25.309565 139761090996096 model_lib_v2.py:652] Step 160400 per-step time 0.248s loss=1.378
    INFO:tensorflow:Step 160500 per-step time 0.260s loss=0.738
    I1010 22:11:50.374689 139761090996096 model_lib_v2.py:652] Step 160500 per-step time 0.260s loss=0.738
    INFO:tensorflow:Step 160600 per-step time 0.242s loss=0.620
    I1010 22:12:15.281212 139761090996096 model_lib_v2.py:652] Step 160600 per-step time 0.242s loss=0.620
    INFO:tensorflow:Step 160700 per-step time 0.250s loss=0.583
    I1010 22:12:40.252692 139761090996096 model_lib_v2.py:652] Step 160700 per-step time 0.250s loss=0.583
    INFO:tensorflow:Step 160800 per-step time 0.251s loss=0.553
    I1010 22:13:05.341211 139761090996096 model_lib_v2.py:652] Step 160800 per-step time 0.251s loss=0.553
    INFO:tensorflow:Step 160900 per-step time 0.256s loss=0.688
    I1010 22:13:30.403152 139761090996096 model_lib_v2.py:652] Step 160900 per-step time 0.256s loss=0.688
    INFO:tensorflow:Step 161000 per-step time 0.243s loss=0.868
    I1010 22:13:55.429965 139761090996096 model_lib_v2.py:652] Step 161000 per-step time 0.243s loss=0.868
    INFO:tensorflow:Step 161100 per-step time 0.247s loss=0.855
    I1010 22:14:21.470018 139761090996096 model_lib_v2.py:652] Step 161100 per-step time 0.247s loss=0.855
    INFO:tensorflow:Step 161200 per-step time 0.246s loss=0.673
    I1010 22:14:46.658047 139761090996096 model_lib_v2.py:652] Step 161200 per-step time 0.246s loss=0.673
    INFO:tensorflow:Step 161300 per-step time 0.238s loss=0.675
    I1010 22:15:11.546081 139761090996096 model_lib_v2.py:652] Step 161300 per-step time 0.238s loss=0.675
    INFO:tensorflow:Step 161400 per-step time 0.249s loss=0.633
    I1010 22:15:36.525322 139761090996096 model_lib_v2.py:652] Step 161400 per-step time 0.249s loss=0.633
    INFO:tensorflow:Step 161500 per-step time 0.254s loss=0.713
    I1010 22:16:01.355353 139761090996096 model_lib_v2.py:652] Step 161500 per-step time 0.254s loss=0.713
    INFO:tensorflow:Step 161600 per-step time 0.259s loss=0.687
    I1010 22:16:26.345951 139761090996096 model_lib_v2.py:652] Step 161600 per-step time 0.259s loss=0.687
    INFO:tensorflow:Step 161700 per-step time 0.245s loss=0.624
    I1010 22:16:51.267565 139761090996096 model_lib_v2.py:652] Step 161700 per-step time 0.245s loss=0.624
    INFO:tensorflow:Step 161800 per-step time 0.248s loss=1.003
    I1010 22:17:16.234725 139761090996096 model_lib_v2.py:652] Step 161800 per-step time 0.248s loss=1.003
    INFO:tensorflow:Step 161900 per-step time 0.247s loss=0.557
    I1010 22:17:41.266870 139761090996096 model_lib_v2.py:652] Step 161900 per-step time 0.247s loss=0.557
    INFO:tensorflow:Step 162000 per-step time 0.240s loss=0.456
    I1010 22:18:06.587154 139761090996096 model_lib_v2.py:652] Step 162000 per-step time 0.240s loss=0.456
    INFO:tensorflow:Step 162100 per-step time 0.259s loss=0.783
    I1010 22:18:32.422891 139761090996096 model_lib_v2.py:652] Step 162100 per-step time 0.259s loss=0.783
    INFO:tensorflow:Step 162200 per-step time 0.240s loss=0.849
    I1010 22:18:57.351094 139761090996096 model_lib_v2.py:652] Step 162200 per-step time 0.240s loss=0.849
    INFO:tensorflow:Step 162300 per-step time 0.259s loss=0.797
    I1010 22:19:22.174523 139761090996096 model_lib_v2.py:652] Step 162300 per-step time 0.259s loss=0.797
    INFO:tensorflow:Step 162400 per-step time 0.238s loss=0.681
    I1010 22:19:46.867997 139761090996096 model_lib_v2.py:652] Step 162400 per-step time 0.238s loss=0.681
    INFO:tensorflow:Step 162500 per-step time 0.238s loss=0.835
    I1010 22:20:11.554926 139761090996096 model_lib_v2.py:652] Step 162500 per-step time 0.238s loss=0.835
    INFO:tensorflow:Step 162600 per-step time 0.255s loss=0.825
    I1010 22:20:36.394266 139761090996096 model_lib_v2.py:652] Step 162600 per-step time 0.255s loss=0.825
    INFO:tensorflow:Step 162700 per-step time 0.240s loss=0.636
    I1010 22:21:01.024586 139761090996096 model_lib_v2.py:652] Step 162700 per-step time 0.240s loss=0.636
    INFO:tensorflow:Step 162800 per-step time 0.242s loss=0.894
    I1010 22:21:25.740095 139761090996096 model_lib_v2.py:652] Step 162800 per-step time 0.242s loss=0.894
    INFO:tensorflow:Step 162900 per-step time 0.247s loss=0.770
    I1010 22:21:50.492889 139761090996096 model_lib_v2.py:652] Step 162900 per-step time 0.247s loss=0.770
    INFO:tensorflow:Step 163000 per-step time 0.242s loss=0.559
    I1010 22:22:15.106261 139761090996096 model_lib_v2.py:652] Step 163000 per-step time 0.242s loss=0.559
    INFO:tensorflow:Step 163100 per-step time 0.238s loss=0.613
    I1010 22:22:40.640464 139761090996096 model_lib_v2.py:652] Step 163100 per-step time 0.238s loss=0.613
    INFO:tensorflow:Step 163200 per-step time 0.247s loss=0.936
    I1010 22:23:05.229646 139761090996096 model_lib_v2.py:652] Step 163200 per-step time 0.247s loss=0.936
    INFO:tensorflow:Step 163300 per-step time 0.246s loss=0.615
    I1010 22:23:29.945335 139761090996096 model_lib_v2.py:652] Step 163300 per-step time 0.246s loss=0.615
    INFO:tensorflow:Step 163400 per-step time 0.254s loss=0.974
    I1010 22:23:54.737945 139761090996096 model_lib_v2.py:652] Step 163400 per-step time 0.254s loss=0.974
    INFO:tensorflow:Step 163500 per-step time 0.252s loss=0.536
    I1010 22:24:19.336110 139761090996096 model_lib_v2.py:652] Step 163500 per-step time 0.252s loss=0.536
    INFO:tensorflow:Step 163600 per-step time 0.247s loss=0.792
    I1010 22:24:43.819196 139761090996096 model_lib_v2.py:652] Step 163600 per-step time 0.247s loss=0.792
    INFO:tensorflow:Step 163700 per-step time 0.241s loss=0.833
    I1010 22:25:08.428646 139761090996096 model_lib_v2.py:652] Step 163700 per-step time 0.241s loss=0.833
    INFO:tensorflow:Step 163800 per-step time 0.240s loss=0.660
    I1010 22:25:33.095533 139761090996096 model_lib_v2.py:652] Step 163800 per-step time 0.240s loss=0.660
    INFO:tensorflow:Step 163900 per-step time 0.259s loss=0.884
    I1010 22:25:57.674353 139761090996096 model_lib_v2.py:652] Step 163900 per-step time 0.259s loss=0.884
    INFO:tensorflow:Step 164000 per-step time 0.243s loss=0.642
    I1010 22:26:22.302799 139761090996096 model_lib_v2.py:652] Step 164000 per-step time 0.243s loss=0.642
    INFO:tensorflow:Step 164100 per-step time 0.247s loss=0.635
    I1010 22:26:47.911509 139761090996096 model_lib_v2.py:652] Step 164100 per-step time 0.247s loss=0.635
    INFO:tensorflow:Step 164200 per-step time 0.262s loss=0.799
    I1010 22:27:12.680653 139761090996096 model_lib_v2.py:652] Step 164200 per-step time 0.262s loss=0.799
    INFO:tensorflow:Step 164300 per-step time 0.237s loss=0.560
    I1010 22:27:37.385357 139761090996096 model_lib_v2.py:652] Step 164300 per-step time 0.237s loss=0.560
    INFO:tensorflow:Step 164400 per-step time 0.243s loss=0.789
    I1010 22:28:02.081601 139761090996096 model_lib_v2.py:652] Step 164400 per-step time 0.243s loss=0.789
    INFO:tensorflow:Step 164500 per-step time 0.260s loss=0.727
    I1010 22:28:26.863035 139761090996096 model_lib_v2.py:652] Step 164500 per-step time 0.260s loss=0.727
    INFO:tensorflow:Step 164600 per-step time 0.260s loss=0.503
    I1010 22:28:51.602090 139761090996096 model_lib_v2.py:652] Step 164600 per-step time 0.260s loss=0.503
    INFO:tensorflow:Step 164700 per-step time 0.243s loss=0.495
    I1010 22:29:16.485008 139761090996096 model_lib_v2.py:652] Step 164700 per-step time 0.243s loss=0.495
    INFO:tensorflow:Step 164800 per-step time 0.248s loss=0.624
    I1010 22:29:41.324528 139761090996096 model_lib_v2.py:652] Step 164800 per-step time 0.248s loss=0.624
    INFO:tensorflow:Step 164900 per-step time 0.248s loss=0.794
    I1010 22:30:06.105970 139761090996096 model_lib_v2.py:652] Step 164900 per-step time 0.248s loss=0.794
    INFO:tensorflow:Step 165000 per-step time 0.246s loss=1.131
    I1010 22:30:30.852324 139761090996096 model_lib_v2.py:652] Step 165000 per-step time 0.246s loss=1.131
    INFO:tensorflow:Step 165100 per-step time 0.240s loss=0.673
    I1010 22:30:56.489153 139761090996096 model_lib_v2.py:652] Step 165100 per-step time 0.240s loss=0.673
    INFO:tensorflow:Step 165200 per-step time 0.241s loss=0.570
    I1010 22:31:21.292886 139761090996096 model_lib_v2.py:652] Step 165200 per-step time 0.241s loss=0.570
    INFO:tensorflow:Step 165300 per-step time 0.259s loss=0.888
    I1010 22:31:46.075430 139761090996096 model_lib_v2.py:652] Step 165300 per-step time 0.259s loss=0.888
    INFO:tensorflow:Step 165400 per-step time 0.276s loss=0.506
    I1010 22:32:10.765520 139761090996096 model_lib_v2.py:652] Step 165400 per-step time 0.276s loss=0.506
    INFO:tensorflow:Step 165500 per-step time 0.246s loss=0.555
    I1010 22:32:35.308494 139761090996096 model_lib_v2.py:652] Step 165500 per-step time 0.246s loss=0.555
    INFO:tensorflow:Step 165600 per-step time 0.248s loss=0.739
    I1010 22:32:59.893237 139761090996096 model_lib_v2.py:652] Step 165600 per-step time 0.248s loss=0.739
    INFO:tensorflow:Step 165700 per-step time 0.263s loss=0.828
    I1010 22:33:24.535166 139761090996096 model_lib_v2.py:652] Step 165700 per-step time 0.263s loss=0.828
    INFO:tensorflow:Step 165800 per-step time 0.253s loss=0.913
    I1010 22:33:49.760437 139761090996096 model_lib_v2.py:652] Step 165800 per-step time 0.253s loss=0.913
    INFO:tensorflow:Step 165900 per-step time 0.247s loss=0.974
    I1010 22:34:14.529602 139761090996096 model_lib_v2.py:652] Step 165900 per-step time 0.247s loss=0.974
    INFO:tensorflow:Step 166000 per-step time 0.244s loss=0.650
    I1010 22:34:39.319853 139761090996096 model_lib_v2.py:652] Step 166000 per-step time 0.244s loss=0.650
    INFO:tensorflow:Step 166100 per-step time 0.250s loss=0.578
    I1010 22:35:04.981790 139761090996096 model_lib_v2.py:652] Step 166100 per-step time 0.250s loss=0.578
    INFO:tensorflow:Step 166200 per-step time 0.243s loss=0.818
    I1010 22:35:29.660410 139761090996096 model_lib_v2.py:652] Step 166200 per-step time 0.243s loss=0.818
    INFO:tensorflow:Step 166300 per-step time 0.246s loss=0.882
    I1010 22:35:54.393639 139761090996096 model_lib_v2.py:652] Step 166300 per-step time 0.246s loss=0.882
    INFO:tensorflow:Step 166400 per-step time 0.250s loss=0.728
    I1010 22:36:19.165248 139761090996096 model_lib_v2.py:652] Step 166400 per-step time 0.250s loss=0.728
    INFO:tensorflow:Step 166500 per-step time 0.244s loss=0.937
    I1010 22:36:43.996216 139761090996096 model_lib_v2.py:652] Step 166500 per-step time 0.244s loss=0.937
    INFO:tensorflow:Step 166600 per-step time 0.248s loss=0.552
    I1010 22:37:08.989072 139761090996096 model_lib_v2.py:652] Step 166600 per-step time 0.248s loss=0.552
    INFO:tensorflow:Step 166700 per-step time 0.247s loss=0.388
    I1010 22:37:33.683349 139761090996096 model_lib_v2.py:652] Step 166700 per-step time 0.247s loss=0.388
    INFO:tensorflow:Step 166800 per-step time 0.249s loss=0.693
    I1010 22:37:58.451277 139761090996096 model_lib_v2.py:652] Step 166800 per-step time 0.249s loss=0.693
    INFO:tensorflow:Step 166900 per-step time 0.241s loss=0.765
    I1010 22:38:23.148448 139761090996096 model_lib_v2.py:652] Step 166900 per-step time 0.241s loss=0.765
    INFO:tensorflow:Step 167000 per-step time 0.244s loss=0.741
    I1010 22:38:47.896056 139761090996096 model_lib_v2.py:652] Step 167000 per-step time 0.244s loss=0.741
    INFO:tensorflow:Step 167100 per-step time 0.243s loss=0.725
    I1010 22:39:13.938089 139761090996096 model_lib_v2.py:652] Step 167100 per-step time 0.243s loss=0.725
    INFO:tensorflow:Step 167200 per-step time 0.253s loss=0.615
    I1010 22:39:38.608029 139761090996096 model_lib_v2.py:652] Step 167200 per-step time 0.253s loss=0.615
    INFO:tensorflow:Step 167300 per-step time 0.247s loss=1.066
    I1010 22:40:03.216962 139761090996096 model_lib_v2.py:652] Step 167300 per-step time 0.247s loss=1.066
    INFO:tensorflow:Step 167400 per-step time 0.252s loss=0.920
    I1010 22:40:27.841886 139761090996096 model_lib_v2.py:652] Step 167400 per-step time 0.252s loss=0.920
    INFO:tensorflow:Step 167500 per-step time 0.246s loss=0.599
    I1010 22:40:52.548859 139761090996096 model_lib_v2.py:652] Step 167500 per-step time 0.246s loss=0.599
    INFO:tensorflow:Step 167600 per-step time 0.251s loss=0.703
    I1010 22:41:17.111486 139761090996096 model_lib_v2.py:652] Step 167600 per-step time 0.251s loss=0.703
    INFO:tensorflow:Step 167700 per-step time 0.252s loss=0.897
    I1010 22:41:41.670734 139761090996096 model_lib_v2.py:652] Step 167700 per-step time 0.252s loss=0.897
    INFO:tensorflow:Step 167800 per-step time 0.258s loss=0.591
    I1010 22:42:06.285253 139761090996096 model_lib_v2.py:652] Step 167800 per-step time 0.258s loss=0.591
    INFO:tensorflow:Step 167900 per-step time 0.247s loss=0.759
    I1010 22:42:30.827494 139761090996096 model_lib_v2.py:652] Step 167900 per-step time 0.247s loss=0.759
    INFO:tensorflow:Step 168000 per-step time 0.248s loss=0.969
    I1010 22:42:55.404411 139761090996096 model_lib_v2.py:652] Step 168000 per-step time 0.248s loss=0.969
    INFO:tensorflow:Step 168100 per-step time 0.232s loss=0.573
    I1010 22:43:21.465308 139761090996096 model_lib_v2.py:652] Step 168100 per-step time 0.232s loss=0.573
    INFO:tensorflow:Step 168200 per-step time 0.245s loss=1.019
    I1010 22:43:46.139661 139761090996096 model_lib_v2.py:652] Step 168200 per-step time 0.245s loss=1.019
    INFO:tensorflow:Step 168300 per-step time 0.248s loss=0.635
    I1010 22:44:10.917374 139761090996096 model_lib_v2.py:652] Step 168300 per-step time 0.248s loss=0.635
    INFO:tensorflow:Step 168400 per-step time 0.259s loss=0.652
    I1010 22:44:35.500696 139761090996096 model_lib_v2.py:652] Step 168400 per-step time 0.259s loss=0.652
    INFO:tensorflow:Step 168500 per-step time 0.249s loss=0.719
    I1010 22:45:00.163128 139761090996096 model_lib_v2.py:652] Step 168500 per-step time 0.249s loss=0.719
    INFO:tensorflow:Step 168600 per-step time 0.239s loss=0.825
    I1010 22:45:24.839291 139761090996096 model_lib_v2.py:652] Step 168600 per-step time 0.239s loss=0.825
    INFO:tensorflow:Step 168700 per-step time 0.238s loss=0.498
    I1010 22:45:49.392679 139761090996096 model_lib_v2.py:652] Step 168700 per-step time 0.238s loss=0.498
    INFO:tensorflow:Step 168800 per-step time 0.241s loss=0.622
    I1010 22:46:13.855495 139761090996096 model_lib_v2.py:652] Step 168800 per-step time 0.241s loss=0.622
    INFO:tensorflow:Step 168900 per-step time 0.239s loss=0.610
    I1010 22:46:38.338079 139761090996096 model_lib_v2.py:652] Step 168900 per-step time 0.239s loss=0.610
    INFO:tensorflow:Step 169000 per-step time 0.250s loss=0.624
    I1010 22:47:02.740993 139761090996096 model_lib_v2.py:652] Step 169000 per-step time 0.250s loss=0.624
    INFO:tensorflow:Step 169100 per-step time 0.237s loss=0.468
    I1010 22:47:28.344781 139761090996096 model_lib_v2.py:652] Step 169100 per-step time 0.237s loss=0.468
    INFO:tensorflow:Step 169200 per-step time 0.244s loss=0.786
    I1010 22:47:52.942798 139761090996096 model_lib_v2.py:652] Step 169200 per-step time 0.244s loss=0.786
    INFO:tensorflow:Step 169300 per-step time 0.240s loss=0.566
    I1010 22:48:17.560846 139761090996096 model_lib_v2.py:652] Step 169300 per-step time 0.240s loss=0.566
    INFO:tensorflow:Step 169400 per-step time 0.251s loss=0.609
    I1010 22:48:42.341069 139761090996096 model_lib_v2.py:652] Step 169400 per-step time 0.251s loss=0.609
    INFO:tensorflow:Step 169500 per-step time 0.254s loss=0.673
    I1010 22:49:07.203435 139761090996096 model_lib_v2.py:652] Step 169500 per-step time 0.254s loss=0.673
    INFO:tensorflow:Step 169600 per-step time 0.238s loss=0.825
    I1010 22:49:32.295973 139761090996096 model_lib_v2.py:652] Step 169600 per-step time 0.238s loss=0.825
    INFO:tensorflow:Step 169700 per-step time 0.250s loss=0.764
    I1010 22:49:57.171644 139761090996096 model_lib_v2.py:652] Step 169700 per-step time 0.250s loss=0.764
    INFO:tensorflow:Step 169800 per-step time 0.248s loss=0.923
    I1010 22:50:22.251634 139761090996096 model_lib_v2.py:652] Step 169800 per-step time 0.248s loss=0.923
    INFO:tensorflow:Step 169900 per-step time 0.247s loss=0.662
    I1010 22:50:47.195788 139761090996096 model_lib_v2.py:652] Step 169900 per-step time 0.247s loss=0.662
    INFO:tensorflow:Step 170000 per-step time 0.276s loss=0.634
    I1010 22:51:12.124370 139761090996096 model_lib_v2.py:652] Step 170000 per-step time 0.276s loss=0.634
    INFO:tensorflow:Step 170100 per-step time 0.237s loss=0.653
    I1010 22:51:37.908092 139761090996096 model_lib_v2.py:652] Step 170100 per-step time 0.237s loss=0.653
    INFO:tensorflow:Step 170200 per-step time 0.250s loss=0.934
    I1010 22:52:02.814552 139761090996096 model_lib_v2.py:652] Step 170200 per-step time 0.250s loss=0.934
    INFO:tensorflow:Step 170300 per-step time 0.234s loss=0.666
    I1010 22:52:27.798699 139761090996096 model_lib_v2.py:652] Step 170300 per-step time 0.234s loss=0.666
    INFO:tensorflow:Step 170400 per-step time 0.250s loss=0.879
    I1010 22:52:52.714401 139761090996096 model_lib_v2.py:652] Step 170400 per-step time 0.250s loss=0.879
    INFO:tensorflow:Step 170500 per-step time 0.243s loss=0.974
    I1010 22:53:17.519676 139761090996096 model_lib_v2.py:652] Step 170500 per-step time 0.243s loss=0.974
    INFO:tensorflow:Step 170600 per-step time 0.261s loss=0.768
    I1010 22:53:42.358913 139761090996096 model_lib_v2.py:652] Step 170600 per-step time 0.261s loss=0.768
    INFO:tensorflow:Step 170700 per-step time 0.255s loss=0.902
    I1010 22:54:07.075151 139761090996096 model_lib_v2.py:652] Step 170700 per-step time 0.255s loss=0.902
    INFO:tensorflow:Step 170800 per-step time 0.239s loss=0.858
    I1010 22:54:32.071919 139761090996096 model_lib_v2.py:652] Step 170800 per-step time 0.239s loss=0.858
    INFO:tensorflow:Step 170900 per-step time 0.252s loss=0.876
    I1010 22:54:56.757844 139761090996096 model_lib_v2.py:652] Step 170900 per-step time 0.252s loss=0.876
    INFO:tensorflow:Step 171000 per-step time 0.261s loss=0.674
    I1010 22:55:21.619934 139761090996096 model_lib_v2.py:652] Step 171000 per-step time 0.261s loss=0.674
    INFO:tensorflow:Step 171100 per-step time 0.258s loss=0.786
    I1010 22:55:47.392273 139761090996096 model_lib_v2.py:652] Step 171100 per-step time 0.258s loss=0.786
    INFO:tensorflow:Step 171200 per-step time 0.263s loss=0.490
    I1010 22:56:12.261877 139761090996096 model_lib_v2.py:652] Step 171200 per-step time 0.263s loss=0.490
    INFO:tensorflow:Step 171300 per-step time 0.260s loss=0.767
    I1010 22:56:37.228331 139761090996096 model_lib_v2.py:652] Step 171300 per-step time 0.260s loss=0.767
    INFO:tensorflow:Step 171400 per-step time 0.258s loss=0.778
    I1010 22:57:02.053834 139761090996096 model_lib_v2.py:652] Step 171400 per-step time 0.258s loss=0.778
    INFO:tensorflow:Step 171500 per-step time 0.255s loss=0.719
    I1010 22:57:26.945490 139761090996096 model_lib_v2.py:652] Step 171500 per-step time 0.255s loss=0.719
    INFO:tensorflow:Step 171600 per-step time 0.249s loss=0.640
    I1010 22:57:51.943936 139761090996096 model_lib_v2.py:652] Step 171600 per-step time 0.249s loss=0.640
    INFO:tensorflow:Step 171700 per-step time 0.252s loss=0.563
    I1010 22:58:16.853883 139761090996096 model_lib_v2.py:652] Step 171700 per-step time 0.252s loss=0.563
    INFO:tensorflow:Step 171800 per-step time 0.245s loss=0.859
    I1010 22:58:41.905977 139761090996096 model_lib_v2.py:652] Step 171800 per-step time 0.245s loss=0.859
    INFO:tensorflow:Step 171900 per-step time 0.248s loss=0.877
    I1010 22:59:06.655178 139761090996096 model_lib_v2.py:652] Step 171900 per-step time 0.248s loss=0.877
    INFO:tensorflow:Step 172000 per-step time 0.245s loss=0.791
    I1010 22:59:31.702421 139761090996096 model_lib_v2.py:652] Step 172000 per-step time 0.245s loss=0.791
    INFO:tensorflow:Step 172100 per-step time 0.250s loss=0.837
    I1010 22:59:57.666414 139761090996096 model_lib_v2.py:652] Step 172100 per-step time 0.250s loss=0.837
    INFO:tensorflow:Step 172200 per-step time 0.246s loss=0.619
    I1010 23:00:22.577654 139761090996096 model_lib_v2.py:652] Step 172200 per-step time 0.246s loss=0.619
    INFO:tensorflow:Step 172300 per-step time 0.262s loss=0.781
    I1010 23:00:47.488826 139761090996096 model_lib_v2.py:652] Step 172300 per-step time 0.262s loss=0.781
    INFO:tensorflow:Step 172400 per-step time 0.240s loss=0.795
    I1010 23:01:12.398056 139761090996096 model_lib_v2.py:652] Step 172400 per-step time 0.240s loss=0.795
    INFO:tensorflow:Step 172500 per-step time 0.233s loss=0.751
    I1010 23:01:37.306003 139761090996096 model_lib_v2.py:652] Step 172500 per-step time 0.233s loss=0.751
    INFO:tensorflow:Step 172600 per-step time 0.253s loss=0.985
    I1010 23:02:02.227918 139761090996096 model_lib_v2.py:652] Step 172600 per-step time 0.253s loss=0.985
    INFO:tensorflow:Step 172700 per-step time 0.254s loss=1.137
    I1010 23:02:27.176108 139761090996096 model_lib_v2.py:652] Step 172700 per-step time 0.254s loss=1.137
    INFO:tensorflow:Step 172800 per-step time 0.240s loss=0.870
    I1010 23:02:52.210994 139761090996096 model_lib_v2.py:652] Step 172800 per-step time 0.240s loss=0.870
    INFO:tensorflow:Step 172900 per-step time 0.241s loss=0.978
    I1010 23:03:17.037237 139761090996096 model_lib_v2.py:652] Step 172900 per-step time 0.241s loss=0.978
    INFO:tensorflow:Step 173000 per-step time 0.250s loss=0.928
    I1010 23:03:41.948037 139761090996096 model_lib_v2.py:652] Step 173000 per-step time 0.250s loss=0.928
    INFO:tensorflow:Step 173100 per-step time 0.257s loss=0.751
    I1010 23:04:07.778479 139761090996096 model_lib_v2.py:652] Step 173100 per-step time 0.257s loss=0.751
    INFO:tensorflow:Step 173200 per-step time 0.239s loss=0.496
    I1010 23:04:32.685809 139761090996096 model_lib_v2.py:652] Step 173200 per-step time 0.239s loss=0.496
    INFO:tensorflow:Step 173300 per-step time 0.255s loss=0.725
    I1010 23:04:57.980433 139761090996096 model_lib_v2.py:652] Step 173300 per-step time 0.255s loss=0.725
    INFO:tensorflow:Step 173400 per-step time 0.236s loss=0.676
    I1010 23:05:22.976410 139761090996096 model_lib_v2.py:652] Step 173400 per-step time 0.236s loss=0.676
    INFO:tensorflow:Step 173500 per-step time 0.252s loss=0.394
    I1010 23:05:47.857174 139761090996096 model_lib_v2.py:652] Step 173500 per-step time 0.252s loss=0.394
    INFO:tensorflow:Step 173600 per-step time 0.246s loss=0.596
    I1010 23:06:12.642206 139761090996096 model_lib_v2.py:652] Step 173600 per-step time 0.246s loss=0.596
    INFO:tensorflow:Step 173700 per-step time 0.253s loss=0.533
    I1010 23:06:37.475633 139761090996096 model_lib_v2.py:652] Step 173700 per-step time 0.253s loss=0.533
    INFO:tensorflow:Step 173800 per-step time 0.248s loss=0.903
    I1010 23:07:02.275515 139761090996096 model_lib_v2.py:652] Step 173800 per-step time 0.248s loss=0.903
    INFO:tensorflow:Step 173900 per-step time 0.254s loss=0.805
    I1010 23:07:27.241099 139761090996096 model_lib_v2.py:652] Step 173900 per-step time 0.254s loss=0.805
    INFO:tensorflow:Step 174000 per-step time 0.235s loss=0.723
    I1010 23:07:52.117859 139761090996096 model_lib_v2.py:652] Step 174000 per-step time 0.235s loss=0.723
    INFO:tensorflow:Step 174100 per-step time 0.255s loss=0.615
    I1010 23:08:17.844133 139761090996096 model_lib_v2.py:652] Step 174100 per-step time 0.255s loss=0.615
    INFO:tensorflow:Step 174200 per-step time 0.239s loss=0.821
    I1010 23:08:42.763499 139761090996096 model_lib_v2.py:652] Step 174200 per-step time 0.239s loss=0.821
    INFO:tensorflow:Step 174300 per-step time 0.235s loss=0.739
    I1010 23:09:07.454327 139761090996096 model_lib_v2.py:652] Step 174300 per-step time 0.235s loss=0.739
    INFO:tensorflow:Step 174400 per-step time 0.239s loss=0.572
    I1010 23:09:32.289194 139761090996096 model_lib_v2.py:652] Step 174400 per-step time 0.239s loss=0.572
    INFO:tensorflow:Step 174500 per-step time 0.252s loss=0.673
    I1010 23:09:57.261673 139761090996096 model_lib_v2.py:652] Step 174500 per-step time 0.252s loss=0.673
    INFO:tensorflow:Step 174600 per-step time 0.257s loss=0.605
    I1010 23:10:22.142047 139761090996096 model_lib_v2.py:652] Step 174600 per-step time 0.257s loss=0.605
    INFO:tensorflow:Step 174700 per-step time 0.248s loss=0.654
    I1010 23:10:47.121944 139761090996096 model_lib_v2.py:652] Step 174700 per-step time 0.248s loss=0.654
    INFO:tensorflow:Step 174800 per-step time 0.236s loss=0.599
    I1010 23:11:11.857793 139761090996096 model_lib_v2.py:652] Step 174800 per-step time 0.236s loss=0.599
    INFO:tensorflow:Step 174900 per-step time 0.261s loss=0.611
    I1010 23:11:36.648170 139761090996096 model_lib_v2.py:652] Step 174900 per-step time 0.261s loss=0.611
    INFO:tensorflow:Step 175000 per-step time 0.248s loss=0.779
    I1010 23:12:01.435846 139761090996096 model_lib_v2.py:652] Step 175000 per-step time 0.248s loss=0.779
    INFO:tensorflow:Step 175100 per-step time 0.241s loss=0.819
    I1010 23:12:27.092031 139761090996096 model_lib_v2.py:652] Step 175100 per-step time 0.241s loss=0.819
    INFO:tensorflow:Step 175200 per-step time 0.242s loss=0.600
    I1010 23:12:52.164014 139761090996096 model_lib_v2.py:652] Step 175200 per-step time 0.242s loss=0.600
    INFO:tensorflow:Step 175300 per-step time 0.255s loss=0.826
    I1010 23:13:17.101402 139761090996096 model_lib_v2.py:652] Step 175300 per-step time 0.255s loss=0.826
    INFO:tensorflow:Step 175400 per-step time 0.269s loss=0.530
    I1010 23:13:41.926288 139761090996096 model_lib_v2.py:652] Step 175400 per-step time 0.269s loss=0.530
    INFO:tensorflow:Step 175500 per-step time 0.261s loss=0.470
    I1010 23:14:06.869776 139761090996096 model_lib_v2.py:652] Step 175500 per-step time 0.261s loss=0.470
    INFO:tensorflow:Step 175600 per-step time 0.248s loss=0.546
    I1010 23:14:31.686136 139761090996096 model_lib_v2.py:652] Step 175600 per-step time 0.248s loss=0.546
    INFO:tensorflow:Step 175700 per-step time 0.253s loss=0.447
    I1010 23:14:56.700343 139761090996096 model_lib_v2.py:652] Step 175700 per-step time 0.253s loss=0.447
    INFO:tensorflow:Step 175800 per-step time 0.242s loss=0.480
    I1010 23:15:21.852798 139761090996096 model_lib_v2.py:652] Step 175800 per-step time 0.242s loss=0.480
    INFO:tensorflow:Step 175900 per-step time 0.255s loss=0.700
    I1010 23:15:46.766414 139761090996096 model_lib_v2.py:652] Step 175900 per-step time 0.255s loss=0.700
    INFO:tensorflow:Step 176000 per-step time 0.267s loss=0.789
    I1010 23:16:11.607677 139761090996096 model_lib_v2.py:652] Step 176000 per-step time 0.267s loss=0.789
    INFO:tensorflow:Step 176100 per-step time 0.253s loss=0.735
    I1010 23:16:37.384760 139761090996096 model_lib_v2.py:652] Step 176100 per-step time 0.253s loss=0.735
    INFO:tensorflow:Step 176200 per-step time 0.261s loss=0.862
    I1010 23:17:02.261613 139761090996096 model_lib_v2.py:652] Step 176200 per-step time 0.261s loss=0.862
    INFO:tensorflow:Step 176300 per-step time 0.262s loss=0.622
    I1010 23:17:27.147134 139761090996096 model_lib_v2.py:652] Step 176300 per-step time 0.262s loss=0.622
    INFO:tensorflow:Step 176400 per-step time 0.252s loss=0.750
    I1010 23:17:51.944736 139761090996096 model_lib_v2.py:652] Step 176400 per-step time 0.252s loss=0.750
    INFO:tensorflow:Step 176500 per-step time 0.233s loss=0.554
    I1010 23:18:16.768009 139761090996096 model_lib_v2.py:652] Step 176500 per-step time 0.233s loss=0.554
    INFO:tensorflow:Step 176600 per-step time 0.241s loss=0.624
    I1010 23:18:41.680183 139761090996096 model_lib_v2.py:652] Step 176600 per-step time 0.241s loss=0.624
    INFO:tensorflow:Step 176700 per-step time 0.240s loss=0.486
    I1010 23:19:06.581466 139761090996096 model_lib_v2.py:652] Step 176700 per-step time 0.240s loss=0.486
    INFO:tensorflow:Step 176800 per-step time 0.254s loss=0.774
    I1010 23:19:31.389638 139761090996096 model_lib_v2.py:652] Step 176800 per-step time 0.254s loss=0.774
    INFO:tensorflow:Step 176900 per-step time 0.244s loss=0.399
    I1010 23:19:56.293792 139761090996096 model_lib_v2.py:652] Step 176900 per-step time 0.244s loss=0.399
    INFO:tensorflow:Step 177000 per-step time 0.254s loss=0.764
    I1010 23:20:21.287797 139761090996096 model_lib_v2.py:652] Step 177000 per-step time 0.254s loss=0.764
    INFO:tensorflow:Step 177100 per-step time 0.251s loss=0.738
    I1010 23:20:47.140310 139761090996096 model_lib_v2.py:652] Step 177100 per-step time 0.251s loss=0.738
    INFO:tensorflow:Step 177200 per-step time 0.258s loss=0.781
    I1010 23:21:12.017984 139761090996096 model_lib_v2.py:652] Step 177200 per-step time 0.258s loss=0.781
    INFO:tensorflow:Step 177300 per-step time 0.235s loss=0.500
    I1010 23:21:36.959684 139761090996096 model_lib_v2.py:652] Step 177300 per-step time 0.235s loss=0.500
    INFO:tensorflow:Step 177400 per-step time 0.249s loss=0.847
    I1010 23:22:01.718723 139761090996096 model_lib_v2.py:652] Step 177400 per-step time 0.249s loss=0.847
    INFO:tensorflow:Step 177500 per-step time 0.269s loss=0.960
    I1010 23:22:26.696601 139761090996096 model_lib_v2.py:652] Step 177500 per-step time 0.269s loss=0.960
    INFO:tensorflow:Step 177600 per-step time 0.244s loss=0.553
    I1010 23:22:51.592567 139761090996096 model_lib_v2.py:652] Step 177600 per-step time 0.244s loss=0.553
    INFO:tensorflow:Step 177700 per-step time 0.238s loss=0.669
    I1010 23:23:16.526771 139761090996096 model_lib_v2.py:652] Step 177700 per-step time 0.238s loss=0.669
    INFO:tensorflow:Step 177800 per-step time 0.241s loss=0.670
    I1010 23:23:41.451681 139761090996096 model_lib_v2.py:652] Step 177800 per-step time 0.241s loss=0.670
    INFO:tensorflow:Step 177900 per-step time 0.252s loss=0.661
    I1010 23:24:06.221146 139761090996096 model_lib_v2.py:652] Step 177900 per-step time 0.252s loss=0.661
    INFO:tensorflow:Step 178000 per-step time 0.242s loss=0.839
    I1010 23:24:31.058849 139761090996096 model_lib_v2.py:652] Step 178000 per-step time 0.242s loss=0.839
    INFO:tensorflow:Step 178100 per-step time 0.254s loss=0.802
    I1010 23:24:56.808788 139761090996096 model_lib_v2.py:652] Step 178100 per-step time 0.254s loss=0.802
    INFO:tensorflow:Step 178200 per-step time 0.253s loss=0.645
    I1010 23:25:21.851592 139761090996096 model_lib_v2.py:652] Step 178200 per-step time 0.253s loss=0.645
    INFO:tensorflow:Step 178300 per-step time 0.246s loss=0.561
    I1010 23:25:46.967598 139761090996096 model_lib_v2.py:652] Step 178300 per-step time 0.246s loss=0.561
    INFO:tensorflow:Step 178400 per-step time 0.242s loss=0.469
    I1010 23:26:11.746994 139761090996096 model_lib_v2.py:652] Step 178400 per-step time 0.242s loss=0.469
    INFO:tensorflow:Step 178500 per-step time 0.245s loss=0.501
    I1010 23:26:36.689104 139761090996096 model_lib_v2.py:652] Step 178500 per-step time 0.245s loss=0.501
    INFO:tensorflow:Step 178600 per-step time 0.250s loss=0.926
    I1010 23:27:01.599193 139761090996096 model_lib_v2.py:652] Step 178600 per-step time 0.250s loss=0.926
    INFO:tensorflow:Step 178700 per-step time 0.255s loss=0.708
    I1010 23:27:26.436750 139761090996096 model_lib_v2.py:652] Step 178700 per-step time 0.255s loss=0.708
    INFO:tensorflow:Step 178800 per-step time 0.239s loss=0.623
    I1010 23:27:51.319173 139761090996096 model_lib_v2.py:652] Step 178800 per-step time 0.239s loss=0.623
    INFO:tensorflow:Step 178900 per-step time 0.243s loss=0.667
    I1010 23:28:16.101132 139761090996096 model_lib_v2.py:652] Step 178900 per-step time 0.243s loss=0.667
    INFO:tensorflow:Step 179000 per-step time 0.258s loss=0.865
    I1010 23:28:41.107785 139761090996096 model_lib_v2.py:652] Step 179000 per-step time 0.258s loss=0.865
    INFO:tensorflow:Step 179100 per-step time 0.237s loss=0.883
    I1010 23:29:06.929553 139761090996096 model_lib_v2.py:652] Step 179100 per-step time 0.237s loss=0.883
    INFO:tensorflow:Step 179200 per-step time 0.263s loss=0.702
    I1010 23:29:31.795725 139761090996096 model_lib_v2.py:652] Step 179200 per-step time 0.263s loss=0.702
    INFO:tensorflow:Step 179300 per-step time 0.250s loss=0.833
    I1010 23:29:56.628774 139761090996096 model_lib_v2.py:652] Step 179300 per-step time 0.250s loss=0.833
    INFO:tensorflow:Step 179400 per-step time 0.243s loss=0.770
    I1010 23:30:21.509405 139761090996096 model_lib_v2.py:652] Step 179400 per-step time 0.243s loss=0.770
    INFO:tensorflow:Step 179500 per-step time 0.256s loss=0.818
    I1010 23:30:46.598046 139761090996096 model_lib_v2.py:652] Step 179500 per-step time 0.256s loss=0.818
    INFO:tensorflow:Step 179600 per-step time 0.262s loss=1.060
    I1010 23:31:11.461963 139761090996096 model_lib_v2.py:652] Step 179600 per-step time 0.262s loss=1.060
    INFO:tensorflow:Step 179700 per-step time 0.242s loss=0.856
    I1010 23:31:36.421489 139761090996096 model_lib_v2.py:652] Step 179700 per-step time 0.242s loss=0.856
    INFO:tensorflow:Step 179800 per-step time 0.250s loss=1.109
    I1010 23:32:01.300319 139761090996096 model_lib_v2.py:652] Step 179800 per-step time 0.250s loss=1.109
    INFO:tensorflow:Step 179900 per-step time 0.252s loss=0.922
    I1010 23:32:26.151963 139761090996096 model_lib_v2.py:652] Step 179900 per-step time 0.252s loss=0.922
    INFO:tensorflow:Step 180000 per-step time 0.253s loss=0.590
    I1010 23:32:51.127545 139761090996096 model_lib_v2.py:652] Step 180000 per-step time 0.253s loss=0.590
    INFO:tensorflow:Step 180100 per-step time 0.238s loss=0.702
    I1010 23:33:16.859310 139761090996096 model_lib_v2.py:652] Step 180100 per-step time 0.238s loss=0.702
    INFO:tensorflow:Step 180200 per-step time 0.243s loss=0.793
    I1010 23:33:41.766916 139761090996096 model_lib_v2.py:652] Step 180200 per-step time 0.243s loss=0.793
    INFO:tensorflow:Step 180300 per-step time 0.249s loss=1.017
    I1010 23:34:06.721778 139761090996096 model_lib_v2.py:652] Step 180300 per-step time 0.249s loss=1.017
    INFO:tensorflow:Step 180400 per-step time 0.250s loss=0.868
    I1010 23:34:31.658109 139761090996096 model_lib_v2.py:652] Step 180400 per-step time 0.250s loss=0.868
    INFO:tensorflow:Step 180500 per-step time 0.259s loss=0.843
    I1010 23:34:56.452044 139761090996096 model_lib_v2.py:652] Step 180500 per-step time 0.259s loss=0.843
    INFO:tensorflow:Step 180600 per-step time 0.245s loss=0.865
    I1010 23:35:21.357043 139761090996096 model_lib_v2.py:652] Step 180600 per-step time 0.245s loss=0.865
    INFO:tensorflow:Step 180700 per-step time 0.261s loss=0.871
    I1010 23:35:46.329607 139761090996096 model_lib_v2.py:652] Step 180700 per-step time 0.261s loss=0.871
    INFO:tensorflow:Step 180800 per-step time 0.241s loss=0.630
    I1010 23:36:11.277995 139761090996096 model_lib_v2.py:652] Step 180800 per-step time 0.241s loss=0.630
    INFO:tensorflow:Step 180900 per-step time 0.241s loss=0.339
    I1010 23:36:36.190638 139761090996096 model_lib_v2.py:652] Step 180900 per-step time 0.241s loss=0.339
    INFO:tensorflow:Step 181000 per-step time 0.252s loss=1.005
    I1010 23:37:01.213446 139761090996096 model_lib_v2.py:652] Step 181000 per-step time 0.252s loss=1.005
    INFO:tensorflow:Step 181100 per-step time 0.241s loss=0.772
    I1010 23:37:27.047517 139761090996096 model_lib_v2.py:652] Step 181100 per-step time 0.241s loss=0.772
    INFO:tensorflow:Step 181200 per-step time 0.247s loss=0.886
    I1010 23:37:52.077845 139761090996096 model_lib_v2.py:652] Step 181200 per-step time 0.247s loss=0.886
    INFO:tensorflow:Step 181300 per-step time 0.248s loss=0.750
    I1010 23:38:17.113795 139761090996096 model_lib_v2.py:652] Step 181300 per-step time 0.248s loss=0.750
    INFO:tensorflow:Step 181400 per-step time 0.265s loss=0.606
    I1010 23:38:42.155895 139761090996096 model_lib_v2.py:652] Step 181400 per-step time 0.265s loss=0.606
    INFO:tensorflow:Step 181500 per-step time 0.256s loss=0.614
    I1010 23:39:07.029987 139761090996096 model_lib_v2.py:652] Step 181500 per-step time 0.256s loss=0.614
    INFO:tensorflow:Step 181600 per-step time 0.255s loss=0.965
    I1010 23:39:31.851516 139761090996096 model_lib_v2.py:652] Step 181600 per-step time 0.255s loss=0.965
    INFO:tensorflow:Step 181700 per-step time 0.250s loss=0.452
    I1010 23:39:56.781157 139761090996096 model_lib_v2.py:652] Step 181700 per-step time 0.250s loss=0.452
    INFO:tensorflow:Step 181800 per-step time 0.259s loss=0.453
    I1010 23:40:21.659409 139761090996096 model_lib_v2.py:652] Step 181800 per-step time 0.259s loss=0.453
    INFO:tensorflow:Step 181900 per-step time 0.249s loss=1.098
    I1010 23:40:46.565625 139761090996096 model_lib_v2.py:652] Step 181900 per-step time 0.249s loss=1.098
    INFO:tensorflow:Step 182000 per-step time 0.242s loss=0.546
    I1010 23:41:11.587130 139761090996096 model_lib_v2.py:652] Step 182000 per-step time 0.242s loss=0.546
    INFO:tensorflow:Step 182100 per-step time 0.248s loss=0.545
    I1010 23:41:37.427886 139761090996096 model_lib_v2.py:652] Step 182100 per-step time 0.248s loss=0.545
    INFO:tensorflow:Step 182200 per-step time 0.253s loss=0.657
    I1010 23:42:02.339732 139761090996096 model_lib_v2.py:652] Step 182200 per-step time 0.253s loss=0.657
    INFO:tensorflow:Step 182300 per-step time 0.239s loss=0.415
    I1010 23:42:27.178373 139761090996096 model_lib_v2.py:652] Step 182300 per-step time 0.239s loss=0.415
    INFO:tensorflow:Step 182400 per-step time 0.267s loss=0.771
    I1010 23:42:52.132842 139761090996096 model_lib_v2.py:652] Step 182400 per-step time 0.267s loss=0.771
    INFO:tensorflow:Step 182500 per-step time 0.251s loss=0.760
    I1010 23:43:17.033607 139761090996096 model_lib_v2.py:652] Step 182500 per-step time 0.251s loss=0.760
    INFO:tensorflow:Step 182600 per-step time 0.239s loss=0.515
    I1010 23:43:41.843393 139761090996096 model_lib_v2.py:652] Step 182600 per-step time 0.239s loss=0.515
    INFO:tensorflow:Step 182700 per-step time 0.248s loss=0.883
    I1010 23:44:06.627875 139761090996096 model_lib_v2.py:652] Step 182700 per-step time 0.248s loss=0.883
    INFO:tensorflow:Step 182800 per-step time 0.246s loss=0.835
    I1010 23:44:31.474830 139761090996096 model_lib_v2.py:652] Step 182800 per-step time 0.246s loss=0.835
    INFO:tensorflow:Step 182900 per-step time 0.248s loss=0.721
    I1010 23:44:56.528765 139761090996096 model_lib_v2.py:652] Step 182900 per-step time 0.248s loss=0.721
    INFO:tensorflow:Step 183000 per-step time 0.249s loss=0.833
    I1010 23:45:21.495642 139761090996096 model_lib_v2.py:652] Step 183000 per-step time 0.249s loss=0.833
    INFO:tensorflow:Step 183100 per-step time 0.248s loss=0.695
    I1010 23:45:47.191230 139761090996096 model_lib_v2.py:652] Step 183100 per-step time 0.248s loss=0.695
    INFO:tensorflow:Step 183200 per-step time 0.244s loss=0.725
    I1010 23:46:12.357017 139761090996096 model_lib_v2.py:652] Step 183200 per-step time 0.244s loss=0.725
    INFO:tensorflow:Step 183300 per-step time 0.243s loss=1.112
    I1010 23:46:37.237546 139761090996096 model_lib_v2.py:652] Step 183300 per-step time 0.243s loss=1.112
    INFO:tensorflow:Step 183400 per-step time 0.242s loss=0.509
    I1010 23:47:02.228090 139761090996096 model_lib_v2.py:652] Step 183400 per-step time 0.242s loss=0.509
    INFO:tensorflow:Step 183500 per-step time 0.257s loss=1.106
    I1010 23:47:27.117788 139761090996096 model_lib_v2.py:652] Step 183500 per-step time 0.257s loss=1.106
    INFO:tensorflow:Step 183600 per-step time 0.254s loss=0.658
    I1010 23:47:52.097114 139761090996096 model_lib_v2.py:652] Step 183600 per-step time 0.254s loss=0.658
    INFO:tensorflow:Step 183700 per-step time 0.279s loss=0.688
    I1010 23:48:17.112872 139761090996096 model_lib_v2.py:652] Step 183700 per-step time 0.279s loss=0.688
    INFO:tensorflow:Step 183800 per-step time 0.246s loss=0.752
    I1010 23:48:42.022896 139761090996096 model_lib_v2.py:652] Step 183800 per-step time 0.246s loss=0.752
    INFO:tensorflow:Step 183900 per-step time 0.251s loss=0.673
    I1010 23:49:06.925276 139761090996096 model_lib_v2.py:652] Step 183900 per-step time 0.251s loss=0.673
    INFO:tensorflow:Step 184000 per-step time 0.234s loss=0.485
    I1010 23:49:31.936096 139761090996096 model_lib_v2.py:652] Step 184000 per-step time 0.234s loss=0.485
    INFO:tensorflow:Step 184100 per-step time 0.261s loss=0.883
    I1010 23:49:57.898865 139761090996096 model_lib_v2.py:652] Step 184100 per-step time 0.261s loss=0.883
    INFO:tensorflow:Step 184200 per-step time 0.247s loss=0.687
    I1010 23:50:22.827959 139761090996096 model_lib_v2.py:652] Step 184200 per-step time 0.247s loss=0.687
    INFO:tensorflow:Step 184300 per-step time 0.243s loss=0.683
    I1010 23:50:47.619231 139761090996096 model_lib_v2.py:652] Step 184300 per-step time 0.243s loss=0.683
    INFO:tensorflow:Step 184400 per-step time 0.239s loss=0.695
    I1010 23:51:12.525381 139761090996096 model_lib_v2.py:652] Step 184400 per-step time 0.239s loss=0.695
    INFO:tensorflow:Step 184500 per-step time 0.248s loss=1.063
    I1010 23:51:37.596270 139761090996096 model_lib_v2.py:652] Step 184500 per-step time 0.248s loss=1.063
    INFO:tensorflow:Step 184600 per-step time 0.252s loss=0.816
    I1010 23:52:02.431765 139761090996096 model_lib_v2.py:652] Step 184600 per-step time 0.252s loss=0.816
    INFO:tensorflow:Step 184700 per-step time 0.241s loss=0.764
    I1010 23:52:27.355117 139761090996096 model_lib_v2.py:652] Step 184700 per-step time 0.241s loss=0.764
    INFO:tensorflow:Step 184800 per-step time 0.248s loss=0.428
    I1010 23:52:52.196767 139761090996096 model_lib_v2.py:652] Step 184800 per-step time 0.248s loss=0.428
    INFO:tensorflow:Step 184900 per-step time 0.243s loss=0.814
    I1010 23:53:17.069284 139761090996096 model_lib_v2.py:652] Step 184900 per-step time 0.243s loss=0.814
    INFO:tensorflow:Step 185000 per-step time 0.252s loss=0.766
    I1010 23:53:41.886259 139761090996096 model_lib_v2.py:652] Step 185000 per-step time 0.252s loss=0.766
    INFO:tensorflow:Step 185100 per-step time 0.239s loss=0.837
    I1010 23:54:07.540117 139761090996096 model_lib_v2.py:652] Step 185100 per-step time 0.239s loss=0.837
    INFO:tensorflow:Step 185200 per-step time 0.252s loss=0.834
    I1010 23:54:32.559824 139761090996096 model_lib_v2.py:652] Step 185200 per-step time 0.252s loss=0.834
    INFO:tensorflow:Step 185300 per-step time 0.260s loss=0.919
    I1010 23:54:57.469423 139761090996096 model_lib_v2.py:652] Step 185300 per-step time 0.260s loss=0.919
    INFO:tensorflow:Step 185400 per-step time 0.246s loss=0.514
    I1010 23:55:22.349168 139761090996096 model_lib_v2.py:652] Step 185400 per-step time 0.246s loss=0.514
    INFO:tensorflow:Step 185500 per-step time 0.242s loss=0.655
    I1010 23:55:47.145042 139761090996096 model_lib_v2.py:652] Step 185500 per-step time 0.242s loss=0.655
    INFO:tensorflow:Step 185600 per-step time 0.246s loss=0.778
    I1010 23:56:12.083831 139761090996096 model_lib_v2.py:652] Step 185600 per-step time 0.246s loss=0.778
    INFO:tensorflow:Step 185700 per-step time 0.254s loss=0.404
    I1010 23:56:37.064605 139761090996096 model_lib_v2.py:652] Step 185700 per-step time 0.254s loss=0.404
    INFO:tensorflow:Step 185800 per-step time 0.244s loss=0.684
    I1010 23:57:01.915252 139761090996096 model_lib_v2.py:652] Step 185800 per-step time 0.244s loss=0.684
    INFO:tensorflow:Step 185900 per-step time 0.258s loss=0.738
    I1010 23:57:26.920839 139761090996096 model_lib_v2.py:652] Step 185900 per-step time 0.258s loss=0.738
    INFO:tensorflow:Step 186000 per-step time 0.258s loss=0.709
    I1010 23:57:51.925854 139761090996096 model_lib_v2.py:652] Step 186000 per-step time 0.258s loss=0.709
    INFO:tensorflow:Step 186100 per-step time 0.258s loss=0.980
    I1010 23:58:17.740800 139761090996096 model_lib_v2.py:652] Step 186100 per-step time 0.258s loss=0.980
    INFO:tensorflow:Step 186200 per-step time 0.252s loss=0.597
    I1010 23:58:42.666791 139761090996096 model_lib_v2.py:652] Step 186200 per-step time 0.252s loss=0.597
    INFO:tensorflow:Step 186300 per-step time 0.235s loss=0.626
    I1010 23:59:07.391205 139761090996096 model_lib_v2.py:652] Step 186300 per-step time 0.235s loss=0.626
    INFO:tensorflow:Step 186400 per-step time 0.237s loss=0.575
    I1010 23:59:32.357086 139761090996096 model_lib_v2.py:652] Step 186400 per-step time 0.237s loss=0.575
    INFO:tensorflow:Step 186500 per-step time 0.249s loss=0.680
    I1010 23:59:57.126796 139761090996096 model_lib_v2.py:652] Step 186500 per-step time 0.249s loss=0.680
    INFO:tensorflow:Step 186600 per-step time 0.246s loss=0.468
    I1011 00:00:21.824997 139761090996096 model_lib_v2.py:652] Step 186600 per-step time 0.246s loss=0.468
    INFO:tensorflow:Step 186700 per-step time 0.239s loss=0.847
    I1011 00:00:46.600296 139761090996096 model_lib_v2.py:652] Step 186700 per-step time 0.239s loss=0.847
    INFO:tensorflow:Step 186800 per-step time 0.248s loss=0.939
    I1011 00:01:11.590761 139761090996096 model_lib_v2.py:652] Step 186800 per-step time 0.248s loss=0.939
    INFO:tensorflow:Step 186900 per-step time 0.252s loss=0.587
    I1011 00:01:36.619317 139761090996096 model_lib_v2.py:652] Step 186900 per-step time 0.252s loss=0.587
    INFO:tensorflow:Step 187000 per-step time 0.234s loss=1.018
    I1011 00:02:01.738261 139761090996096 model_lib_v2.py:652] Step 187000 per-step time 0.234s loss=1.018
    INFO:tensorflow:Step 187100 per-step time 0.255s loss=0.911
    I1011 00:02:27.643247 139761090996096 model_lib_v2.py:652] Step 187100 per-step time 0.255s loss=0.911
    INFO:tensorflow:Step 187200 per-step time 0.245s loss=0.662
    I1011 00:02:52.559228 139761090996096 model_lib_v2.py:652] Step 187200 per-step time 0.245s loss=0.662
    INFO:tensorflow:Step 187300 per-step time 0.239s loss=0.593
    I1011 00:03:17.577775 139761090996096 model_lib_v2.py:652] Step 187300 per-step time 0.239s loss=0.593
    INFO:tensorflow:Step 187400 per-step time 0.250s loss=0.621
    I1011 00:03:42.584044 139761090996096 model_lib_v2.py:652] Step 187400 per-step time 0.250s loss=0.621
    INFO:tensorflow:Step 187500 per-step time 0.249s loss=0.606
    I1011 00:04:07.357640 139761090996096 model_lib_v2.py:652] Step 187500 per-step time 0.249s loss=0.606
    INFO:tensorflow:Step 187600 per-step time 0.258s loss=0.612
    I1011 00:04:32.340162 139761090996096 model_lib_v2.py:652] Step 187600 per-step time 0.258s loss=0.612
    INFO:tensorflow:Step 187700 per-step time 0.244s loss=0.499
    I1011 00:04:57.158364 139761090996096 model_lib_v2.py:652] Step 187700 per-step time 0.244s loss=0.499
    INFO:tensorflow:Step 187800 per-step time 0.255s loss=0.663
    I1011 00:05:21.981232 139761090996096 model_lib_v2.py:652] Step 187800 per-step time 0.255s loss=0.663
    INFO:tensorflow:Step 187900 per-step time 0.243s loss=0.739
    I1011 00:05:46.974245 139761090996096 model_lib_v2.py:652] Step 187900 per-step time 0.243s loss=0.739
    INFO:tensorflow:Step 188000 per-step time 0.256s loss=0.833
    I1011 00:06:11.814426 139761090996096 model_lib_v2.py:652] Step 188000 per-step time 0.256s loss=0.833
    INFO:tensorflow:Step 188100 per-step time 0.233s loss=0.914
    I1011 00:06:37.703728 139761090996096 model_lib_v2.py:652] Step 188100 per-step time 0.233s loss=0.914
    INFO:tensorflow:Step 188200 per-step time 0.248s loss=0.689
    I1011 00:07:02.765552 139761090996096 model_lib_v2.py:652] Step 188200 per-step time 0.248s loss=0.689
    INFO:tensorflow:Step 188300 per-step time 0.255s loss=0.673
    I1011 00:07:27.676796 139761090996096 model_lib_v2.py:652] Step 188300 per-step time 0.255s loss=0.673
    INFO:tensorflow:Step 188400 per-step time 0.248s loss=0.682
    I1011 00:07:52.585079 139761090996096 model_lib_v2.py:652] Step 188400 per-step time 0.248s loss=0.682
    INFO:tensorflow:Step 188500 per-step time 0.236s loss=0.708
    I1011 00:08:17.520848 139761090996096 model_lib_v2.py:652] Step 188500 per-step time 0.236s loss=0.708
    INFO:tensorflow:Step 188600 per-step time 0.264s loss=0.764
    I1011 00:08:42.352770 139761090996096 model_lib_v2.py:652] Step 188600 per-step time 0.264s loss=0.764
    INFO:tensorflow:Step 188700 per-step time 0.233s loss=0.934
    I1011 00:09:07.221275 139761090996096 model_lib_v2.py:652] Step 188700 per-step time 0.233s loss=0.934
    INFO:tensorflow:Step 188800 per-step time 0.256s loss=0.659
    I1011 00:09:32.200354 139761090996096 model_lib_v2.py:652] Step 188800 per-step time 0.256s loss=0.659
    INFO:tensorflow:Step 188900 per-step time 0.256s loss=0.872
    I1011 00:09:57.139538 139761090996096 model_lib_v2.py:652] Step 188900 per-step time 0.256s loss=0.872
    INFO:tensorflow:Step 189000 per-step time 0.243s loss=0.889
    I1011 00:10:22.099667 139761090996096 model_lib_v2.py:652] Step 189000 per-step time 0.243s loss=0.889
    INFO:tensorflow:Step 189100 per-step time 0.243s loss=0.558
    I1011 00:10:48.001424 139761090996096 model_lib_v2.py:652] Step 189100 per-step time 0.243s loss=0.558
    INFO:tensorflow:Step 189200 per-step time 0.253s loss=0.715
    I1011 00:11:13.015286 139761090996096 model_lib_v2.py:652] Step 189200 per-step time 0.253s loss=0.715
    INFO:tensorflow:Step 189300 per-step time 0.252s loss=0.588
    I1011 00:11:38.033355 139761090996096 model_lib_v2.py:652] Step 189300 per-step time 0.252s loss=0.588
    INFO:tensorflow:Step 189400 per-step time 0.254s loss=0.955
    I1011 00:12:02.992785 139761090996096 model_lib_v2.py:652] Step 189400 per-step time 0.254s loss=0.955
    INFO:tensorflow:Step 189500 per-step time 0.263s loss=0.617
    I1011 00:12:28.005620 139761090996096 model_lib_v2.py:652] Step 189500 per-step time 0.263s loss=0.617
    INFO:tensorflow:Step 189600 per-step time 0.247s loss=0.755
    I1011 00:12:52.888519 139761090996096 model_lib_v2.py:652] Step 189600 per-step time 0.247s loss=0.755
    INFO:tensorflow:Step 189700 per-step time 0.236s loss=0.694
    I1011 00:13:17.774376 139761090996096 model_lib_v2.py:652] Step 189700 per-step time 0.236s loss=0.694
    INFO:tensorflow:Step 189800 per-step time 0.244s loss=0.510
    I1011 00:13:42.631895 139761090996096 model_lib_v2.py:652] Step 189800 per-step time 0.244s loss=0.510
    INFO:tensorflow:Step 189900 per-step time 0.245s loss=0.933
    I1011 00:14:07.470252 139761090996096 model_lib_v2.py:652] Step 189900 per-step time 0.245s loss=0.933
    INFO:tensorflow:Step 190000 per-step time 0.246s loss=0.492
    I1011 00:14:32.417227 139761090996096 model_lib_v2.py:652] Step 190000 per-step time 0.246s loss=0.492
    INFO:tensorflow:Step 190100 per-step time 0.239s loss=0.602
    I1011 00:14:58.210875 139761090996096 model_lib_v2.py:652] Step 190100 per-step time 0.239s loss=0.602
    INFO:tensorflow:Step 190200 per-step time 0.240s loss=0.859
    I1011 00:15:23.080680 139761090996096 model_lib_v2.py:652] Step 190200 per-step time 0.240s loss=0.859
    INFO:tensorflow:Step 190300 per-step time 0.244s loss=0.816
    I1011 00:15:48.081293 139761090996096 model_lib_v2.py:652] Step 190300 per-step time 0.244s loss=0.816
    INFO:tensorflow:Step 190400 per-step time 0.242s loss=0.667
    I1011 00:16:12.863207 139761090996096 model_lib_v2.py:652] Step 190400 per-step time 0.242s loss=0.667
    INFO:tensorflow:Step 190500 per-step time 0.238s loss=0.650
    I1011 00:16:37.779054 139761090996096 model_lib_v2.py:652] Step 190500 per-step time 0.238s loss=0.650
    INFO:tensorflow:Step 190600 per-step time 0.245s loss=0.799
    I1011 00:17:02.713098 139761090996096 model_lib_v2.py:652] Step 190600 per-step time 0.245s loss=0.799
    INFO:tensorflow:Step 190700 per-step time 0.245s loss=0.756
    I1011 00:17:27.819800 139761090996096 model_lib_v2.py:652] Step 190700 per-step time 0.245s loss=0.756
    INFO:tensorflow:Step 190800 per-step time 0.254s loss=0.854
    I1011 00:17:52.808935 139761090996096 model_lib_v2.py:652] Step 190800 per-step time 0.254s loss=0.854
    INFO:tensorflow:Step 190900 per-step time 0.250s loss=0.357
    I1011 00:18:17.764025 139761090996096 model_lib_v2.py:652] Step 190900 per-step time 0.250s loss=0.357
    INFO:tensorflow:Step 191000 per-step time 0.254s loss=0.713
    I1011 00:18:42.764646 139761090996096 model_lib_v2.py:652] Step 191000 per-step time 0.254s loss=0.713
    INFO:tensorflow:Step 191100 per-step time 0.257s loss=0.544
    I1011 00:19:08.656190 139761090996096 model_lib_v2.py:652] Step 191100 per-step time 0.257s loss=0.544
    INFO:tensorflow:Step 191200 per-step time 0.256s loss=0.723
    I1011 00:19:33.578588 139761090996096 model_lib_v2.py:652] Step 191200 per-step time 0.256s loss=0.723
    INFO:tensorflow:Step 191300 per-step time 0.243s loss=0.947
    I1011 00:19:58.590934 139761090996096 model_lib_v2.py:652] Step 191300 per-step time 0.243s loss=0.947
    INFO:tensorflow:Step 191400 per-step time 0.259s loss=0.686
    I1011 00:20:23.372455 139761090996096 model_lib_v2.py:652] Step 191400 per-step time 0.259s loss=0.686
    INFO:tensorflow:Step 191500 per-step time 0.245s loss=0.512
    I1011 00:20:48.227563 139761090996096 model_lib_v2.py:652] Step 191500 per-step time 0.245s loss=0.512
    INFO:tensorflow:Step 191600 per-step time 0.237s loss=0.864
    I1011 00:21:13.210573 139761090996096 model_lib_v2.py:652] Step 191600 per-step time 0.237s loss=0.864
    INFO:tensorflow:Step 191700 per-step time 0.241s loss=0.650
    I1011 00:21:38.263380 139761090996096 model_lib_v2.py:652] Step 191700 per-step time 0.241s loss=0.650
    INFO:tensorflow:Step 191800 per-step time 0.246s loss=0.856
    I1011 00:22:03.087795 139761090996096 model_lib_v2.py:652] Step 191800 per-step time 0.246s loss=0.856
    INFO:tensorflow:Step 191900 per-step time 0.252s loss=0.574
    I1011 00:22:28.043775 139761090996096 model_lib_v2.py:652] Step 191900 per-step time 0.252s loss=0.574
    INFO:tensorflow:Step 192000 per-step time 0.249s loss=0.516
    I1011 00:22:53.182065 139761090996096 model_lib_v2.py:652] Step 192000 per-step time 0.249s loss=0.516
    INFO:tensorflow:Step 192100 per-step time 0.248s loss=0.663
    I1011 00:23:19.054667 139761090996096 model_lib_v2.py:652] Step 192100 per-step time 0.248s loss=0.663
    INFO:tensorflow:Step 192200 per-step time 0.251s loss=0.640
    I1011 00:23:43.861600 139761090996096 model_lib_v2.py:652] Step 192200 per-step time 0.251s loss=0.640
    INFO:tensorflow:Step 192300 per-step time 0.245s loss=0.503
    I1011 00:24:08.755916 139761090996096 model_lib_v2.py:652] Step 192300 per-step time 0.245s loss=0.503
    INFO:tensorflow:Step 192400 per-step time 0.259s loss=0.702
    I1011 00:24:33.682385 139761090996096 model_lib_v2.py:652] Step 192400 per-step time 0.259s loss=0.702
    INFO:tensorflow:Step 192500 per-step time 0.257s loss=0.774
    I1011 00:24:58.585665 139761090996096 model_lib_v2.py:652] Step 192500 per-step time 0.257s loss=0.774
    INFO:tensorflow:Step 192600 per-step time 0.236s loss=0.722
    I1011 00:25:23.411647 139761090996096 model_lib_v2.py:652] Step 192600 per-step time 0.236s loss=0.722
    INFO:tensorflow:Step 192700 per-step time 0.243s loss=0.505
    I1011 00:25:48.274157 139761090996096 model_lib_v2.py:652] Step 192700 per-step time 0.243s loss=0.505
    INFO:tensorflow:Step 192800 per-step time 0.255s loss=0.678
    I1011 00:26:13.134602 139761090996096 model_lib_v2.py:652] Step 192800 per-step time 0.255s loss=0.678
    INFO:tensorflow:Step 192900 per-step time 0.246s loss=0.728
    I1011 00:26:38.029833 139761090996096 model_lib_v2.py:652] Step 192900 per-step time 0.246s loss=0.728
    INFO:tensorflow:Step 193000 per-step time 0.247s loss=0.709
    I1011 00:27:02.891725 139761090996096 model_lib_v2.py:652] Step 193000 per-step time 0.247s loss=0.709
    INFO:tensorflow:Step 193100 per-step time 0.250s loss=0.505
    I1011 00:27:28.592044 139761090996096 model_lib_v2.py:652] Step 193100 per-step time 0.250s loss=0.505
    INFO:tensorflow:Step 193200 per-step time 0.243s loss=0.677
    I1011 00:27:53.695016 139761090996096 model_lib_v2.py:652] Step 193200 per-step time 0.243s loss=0.677
    INFO:tensorflow:Step 193300 per-step time 0.242s loss=0.746
    I1011 00:28:18.648353 139761090996096 model_lib_v2.py:652] Step 193300 per-step time 0.242s loss=0.746
    INFO:tensorflow:Step 193400 per-step time 0.249s loss=0.824
    I1011 00:28:43.552205 139761090996096 model_lib_v2.py:652] Step 193400 per-step time 0.249s loss=0.824
    INFO:tensorflow:Step 193500 per-step time 0.243s loss=0.775
    I1011 00:29:08.424459 139761090996096 model_lib_v2.py:652] Step 193500 per-step time 0.243s loss=0.775
    INFO:tensorflow:Step 193600 per-step time 0.241s loss=0.673
    I1011 00:29:33.398102 139761090996096 model_lib_v2.py:652] Step 193600 per-step time 0.241s loss=0.673
    INFO:tensorflow:Step 193700 per-step time 0.254s loss=0.673
    I1011 00:29:58.214616 139761090996096 model_lib_v2.py:652] Step 193700 per-step time 0.254s loss=0.673
    INFO:tensorflow:Step 193800 per-step time 0.271s loss=0.739
    I1011 00:30:23.191689 139761090996096 model_lib_v2.py:652] Step 193800 per-step time 0.271s loss=0.739
    INFO:tensorflow:Step 193900 per-step time 0.249s loss=0.793
    I1011 00:30:48.043333 139761090996096 model_lib_v2.py:652] Step 193900 per-step time 0.249s loss=0.793
    INFO:tensorflow:Step 194000 per-step time 0.256s loss=0.757
    I1011 00:31:13.031049 139761090996096 model_lib_v2.py:652] Step 194000 per-step time 0.256s loss=0.757
    INFO:tensorflow:Step 194100 per-step time 0.247s loss=0.695
    I1011 00:31:38.756280 139761090996096 model_lib_v2.py:652] Step 194100 per-step time 0.247s loss=0.695
    INFO:tensorflow:Step 194200 per-step time 0.249s loss=0.487
    I1011 00:32:03.561352 139761090996096 model_lib_v2.py:652] Step 194200 per-step time 0.249s loss=0.487
    INFO:tensorflow:Step 194300 per-step time 0.250s loss=0.670
    I1011 00:32:28.571372 139761090996096 model_lib_v2.py:652] Step 194300 per-step time 0.250s loss=0.670
    INFO:tensorflow:Step 194400 per-step time 0.252s loss=0.650
    I1011 00:32:53.494513 139761090996096 model_lib_v2.py:652] Step 194400 per-step time 0.252s loss=0.650
    INFO:tensorflow:Step 194500 per-step time 0.250s loss=0.597
    I1011 00:33:18.400992 139761090996096 model_lib_v2.py:652] Step 194500 per-step time 0.250s loss=0.597
    INFO:tensorflow:Step 194600 per-step time 0.240s loss=0.530
    I1011 00:33:43.203549 139761090996096 model_lib_v2.py:652] Step 194600 per-step time 0.240s loss=0.530
    INFO:tensorflow:Step 194700 per-step time 0.251s loss=0.595
    I1011 00:34:08.050332 139761090996096 model_lib_v2.py:652] Step 194700 per-step time 0.251s loss=0.595
    INFO:tensorflow:Step 194800 per-step time 0.252s loss=0.810
    I1011 00:34:32.808551 139761090996096 model_lib_v2.py:652] Step 194800 per-step time 0.252s loss=0.810
    INFO:tensorflow:Step 194900 per-step time 0.246s loss=1.042
    I1011 00:34:57.644222 139761090996096 model_lib_v2.py:652] Step 194900 per-step time 0.246s loss=1.042
    INFO:tensorflow:Step 195000 per-step time 0.240s loss=0.688
    I1011 00:35:22.614173 139761090996096 model_lib_v2.py:652] Step 195000 per-step time 0.240s loss=0.688
    INFO:tensorflow:Step 195100 per-step time 0.259s loss=0.827
    I1011 00:35:48.377641 139761090996096 model_lib_v2.py:652] Step 195100 per-step time 0.259s loss=0.827
    INFO:tensorflow:Step 195200 per-step time 0.244s loss=0.496
    I1011 00:36:13.264471 139761090996096 model_lib_v2.py:652] Step 195200 per-step time 0.244s loss=0.496
    INFO:tensorflow:Step 195300 per-step time 0.249s loss=0.789
    I1011 00:36:38.064368 139761090996096 model_lib_v2.py:652] Step 195300 per-step time 0.249s loss=0.789
    INFO:tensorflow:Step 195400 per-step time 0.264s loss=0.536
    I1011 00:37:02.914232 139761090996096 model_lib_v2.py:652] Step 195400 per-step time 0.264s loss=0.536
    INFO:tensorflow:Step 195500 per-step time 0.243s loss=0.616
    I1011 00:37:27.775273 139761090996096 model_lib_v2.py:652] Step 195500 per-step time 0.243s loss=0.616
    INFO:tensorflow:Step 195600 per-step time 0.253s loss=0.454
    I1011 00:37:52.650947 139761090996096 model_lib_v2.py:652] Step 195600 per-step time 0.253s loss=0.454
    INFO:tensorflow:Step 195700 per-step time 0.252s loss=0.838
    I1011 00:38:17.614050 139761090996096 model_lib_v2.py:652] Step 195700 per-step time 0.252s loss=0.838
    INFO:tensorflow:Step 195800 per-step time 0.262s loss=0.655
    I1011 00:38:42.397731 139761090996096 model_lib_v2.py:652] Step 195800 per-step time 0.262s loss=0.655
    INFO:tensorflow:Step 195900 per-step time 0.242s loss=0.656
    I1011 00:39:07.283425 139761090996096 model_lib_v2.py:652] Step 195900 per-step time 0.242s loss=0.656
    INFO:tensorflow:Step 196000 per-step time 0.252s loss=0.945
    I1011 00:39:32.153671 139761090996096 model_lib_v2.py:652] Step 196000 per-step time 0.252s loss=0.945
    INFO:tensorflow:Step 196100 per-step time 0.252s loss=0.659
    I1011 00:39:57.888030 139761090996096 model_lib_v2.py:652] Step 196100 per-step time 0.252s loss=0.659
    INFO:tensorflow:Step 196200 per-step time 0.238s loss=0.662
    I1011 00:40:22.834287 139761090996096 model_lib_v2.py:652] Step 196200 per-step time 0.238s loss=0.662
    INFO:tensorflow:Step 196300 per-step time 0.261s loss=0.651
    I1011 00:40:47.614521 139761090996096 model_lib_v2.py:652] Step 196300 per-step time 0.261s loss=0.651
    INFO:tensorflow:Step 196400 per-step time 0.245s loss=0.660
    I1011 00:41:12.465953 139761090996096 model_lib_v2.py:652] Step 196400 per-step time 0.245s loss=0.660
    INFO:tensorflow:Step 196500 per-step time 0.252s loss=0.387
    I1011 00:41:37.372957 139761090996096 model_lib_v2.py:652] Step 196500 per-step time 0.252s loss=0.387
    INFO:tensorflow:Step 196600 per-step time 0.249s loss=0.743
    I1011 00:42:02.117353 139761090996096 model_lib_v2.py:652] Step 196600 per-step time 0.249s loss=0.743
    INFO:tensorflow:Step 196700 per-step time 0.237s loss=0.599
    I1011 00:42:26.947496 139761090996096 model_lib_v2.py:652] Step 196700 per-step time 0.237s loss=0.599
    INFO:tensorflow:Step 196800 per-step time 0.252s loss=1.030
    I1011 00:42:51.749673 139761090996096 model_lib_v2.py:652] Step 196800 per-step time 0.252s loss=1.030
    INFO:tensorflow:Step 196900 per-step time 0.257s loss=0.489
    I1011 00:43:16.668419 139761090996096 model_lib_v2.py:652] Step 196900 per-step time 0.257s loss=0.489
    INFO:tensorflow:Step 197000 per-step time 0.255s loss=0.411
    I1011 00:43:41.550423 139761090996096 model_lib_v2.py:652] Step 197000 per-step time 0.255s loss=0.411
    INFO:tensorflow:Step 197100 per-step time 0.260s loss=0.743
    I1011 00:44:07.348957 139761090996096 model_lib_v2.py:652] Step 197100 per-step time 0.260s loss=0.743
    INFO:tensorflow:Step 197200 per-step time 0.248s loss=0.603
    I1011 00:44:32.206929 139761090996096 model_lib_v2.py:652] Step 197200 per-step time 0.248s loss=0.603
    INFO:tensorflow:Step 197300 per-step time 0.240s loss=0.628
    I1011 00:44:57.042058 139761090996096 model_lib_v2.py:652] Step 197300 per-step time 0.240s loss=0.628
    INFO:tensorflow:Step 197400 per-step time 0.234s loss=1.140
    I1011 00:45:21.848496 139761090996096 model_lib_v2.py:652] Step 197400 per-step time 0.234s loss=1.140
    INFO:tensorflow:Step 197500 per-step time 0.247s loss=0.573
    I1011 00:45:46.687286 139761090996096 model_lib_v2.py:652] Step 197500 per-step time 0.247s loss=0.573
    INFO:tensorflow:Step 197600 per-step time 0.259s loss=0.504
    I1011 00:46:11.561031 139761090996096 model_lib_v2.py:652] Step 197600 per-step time 0.259s loss=0.504
    INFO:tensorflow:Step 197700 per-step time 0.245s loss=0.629
    I1011 00:46:36.304856 139761090996096 model_lib_v2.py:652] Step 197700 per-step time 0.245s loss=0.629
    INFO:tensorflow:Step 197800 per-step time 0.258s loss=0.556
    I1011 00:47:01.131374 139761090996096 model_lib_v2.py:652] Step 197800 per-step time 0.258s loss=0.556
    INFO:tensorflow:Step 197900 per-step time 0.248s loss=0.770
    I1011 00:47:25.911069 139761090996096 model_lib_v2.py:652] Step 197900 per-step time 0.248s loss=0.770
    INFO:tensorflow:Step 198000 per-step time 0.256s loss=0.679
    I1011 00:47:50.616748 139761090996096 model_lib_v2.py:652] Step 198000 per-step time 0.256s loss=0.679
    INFO:tensorflow:Step 198100 per-step time 0.244s loss=0.589
    I1011 00:48:16.412137 139761090996096 model_lib_v2.py:652] Step 198100 per-step time 0.244s loss=0.589
    INFO:tensorflow:Step 198200 per-step time 0.245s loss=0.470
    I1011 00:48:41.426875 139761090996096 model_lib_v2.py:652] Step 198200 per-step time 0.245s loss=0.470
    INFO:tensorflow:Step 198300 per-step time 0.245s loss=0.719
    I1011 00:49:06.306234 139761090996096 model_lib_v2.py:652] Step 198300 per-step time 0.245s loss=0.719
    INFO:tensorflow:Step 198400 per-step time 0.243s loss=0.789
    I1011 00:49:31.207983 139761090996096 model_lib_v2.py:652] Step 198400 per-step time 0.243s loss=0.789
    INFO:tensorflow:Step 198500 per-step time 0.249s loss=0.720
    I1011 00:49:56.037866 139761090996096 model_lib_v2.py:652] Step 198500 per-step time 0.249s loss=0.720
    INFO:tensorflow:Step 198600 per-step time 0.252s loss=0.782
    I1011 00:50:21.027147 139761090996096 model_lib_v2.py:652] Step 198600 per-step time 0.252s loss=0.782
    INFO:tensorflow:Step 198700 per-step time 0.265s loss=0.611
    I1011 00:50:45.887514 139761090996096 model_lib_v2.py:652] Step 198700 per-step time 0.265s loss=0.611
    INFO:tensorflow:Step 198800 per-step time 0.253s loss=0.502
    I1011 00:51:10.747637 139761090996096 model_lib_v2.py:652] Step 198800 per-step time 0.253s loss=0.502
    INFO:tensorflow:Step 198900 per-step time 0.245s loss=0.537
    I1011 00:51:35.494971 139761090996096 model_lib_v2.py:652] Step 198900 per-step time 0.245s loss=0.537
    INFO:tensorflow:Step 199000 per-step time 0.247s loss=0.485
    I1011 00:52:00.358091 139761090996096 model_lib_v2.py:652] Step 199000 per-step time 0.247s loss=0.485
    INFO:tensorflow:Step 199100 per-step time 0.252s loss=0.551
    I1011 00:52:26.232159 139761090996096 model_lib_v2.py:652] Step 199100 per-step time 0.252s loss=0.551
    INFO:tensorflow:Step 199200 per-step time 0.248s loss=0.683
    I1011 00:52:51.091404 139761090996096 model_lib_v2.py:652] Step 199200 per-step time 0.248s loss=0.683
    INFO:tensorflow:Step 199300 per-step time 0.245s loss=0.763
    I1011 00:53:15.923530 139761090996096 model_lib_v2.py:652] Step 199300 per-step time 0.245s loss=0.763
    INFO:tensorflow:Step 199400 per-step time 0.258s loss=0.781
    I1011 00:53:41.085846 139761090996096 model_lib_v2.py:652] Step 199400 per-step time 0.258s loss=0.781
    INFO:tensorflow:Step 199500 per-step time 0.234s loss=0.737
    I1011 00:54:05.855532 139761090996096 model_lib_v2.py:652] Step 199500 per-step time 0.234s loss=0.737
    INFO:tensorflow:Step 199600 per-step time 0.245s loss=1.018
    I1011 00:54:30.575404 139761090996096 model_lib_v2.py:652] Step 199600 per-step time 0.245s loss=1.018
    INFO:tensorflow:Step 199700 per-step time 0.258s loss=0.509
    I1011 00:54:55.284580 139761090996096 model_lib_v2.py:652] Step 199700 per-step time 0.258s loss=0.509
    INFO:tensorflow:Step 199800 per-step time 0.263s loss=0.603
    I1011 00:55:20.148412 139761090996096 model_lib_v2.py:652] Step 199800 per-step time 0.263s loss=0.603
    INFO:tensorflow:Step 199900 per-step time 0.256s loss=0.476
    I1011 00:55:45.058745 139761090996096 model_lib_v2.py:652] Step 199900 per-step time 0.256s loss=0.476
    INFO:tensorflow:Step 200000 per-step time 0.252s loss=0.571
    I1011 00:56:09.941445 139761090996096 model_lib_v2.py:652] Step 200000 per-step time 0.252s loss=0.571
    INFO:tensorflow:Step 200100 per-step time 0.241s loss=0.648
    I1011 00:56:35.778561 139761090996096 model_lib_v2.py:652] Step 200100 per-step time 0.241s loss=0.648
    INFO:tensorflow:Step 200200 per-step time 0.236s loss=0.836
    I1011 00:57:00.556372 139761090996096 model_lib_v2.py:652] Step 200200 per-step time 0.236s loss=0.836
    INFO:tensorflow:Step 200300 per-step time 0.262s loss=0.789
    I1011 00:57:25.457462 139761090996096 model_lib_v2.py:652] Step 200300 per-step time 0.262s loss=0.789
    INFO:tensorflow:Step 200400 per-step time 0.267s loss=0.695
    I1011 00:57:50.391933 139761090996096 model_lib_v2.py:652] Step 200400 per-step time 0.267s loss=0.695
    INFO:tensorflow:Step 200500 per-step time 0.262s loss=0.778
    I1011 00:58:15.329952 139761090996096 model_lib_v2.py:652] Step 200500 per-step time 0.262s loss=0.778
    INFO:tensorflow:Step 200600 per-step time 0.243s loss=0.819
    I1011 00:58:40.121457 139761090996096 model_lib_v2.py:652] Step 200600 per-step time 0.243s loss=0.819
    INFO:tensorflow:Step 200700 per-step time 0.252s loss=0.456
    I1011 00:59:05.318629 139761090996096 model_lib_v2.py:652] Step 200700 per-step time 0.252s loss=0.456
    INFO:tensorflow:Step 200800 per-step time 0.241s loss=0.515
    I1011 00:59:30.115005 139761090996096 model_lib_v2.py:652] Step 200800 per-step time 0.241s loss=0.515
    INFO:tensorflow:Step 200900 per-step time 0.245s loss=0.662
    I1011 00:59:55.128976 139761090996096 model_lib_v2.py:652] Step 200900 per-step time 0.245s loss=0.662
    INFO:tensorflow:Step 201000 per-step time 0.253s loss=0.621
    I1011 01:00:19.986153 139761090996096 model_lib_v2.py:652] Step 201000 per-step time 0.253s loss=0.621
    INFO:tensorflow:Step 201100 per-step time 0.261s loss=0.684
    I1011 01:00:45.789536 139761090996096 model_lib_v2.py:652] Step 201100 per-step time 0.261s loss=0.684
    INFO:tensorflow:Step 201200 per-step time 0.257s loss=0.519
    I1011 01:01:10.675470 139761090996096 model_lib_v2.py:652] Step 201200 per-step time 0.257s loss=0.519
    INFO:tensorflow:Step 201300 per-step time 0.249s loss=0.609
    I1011 01:01:35.525138 139761090996096 model_lib_v2.py:652] Step 201300 per-step time 0.249s loss=0.609
    INFO:tensorflow:Step 201400 per-step time 0.240s loss=0.544
    I1011 01:02:00.320415 139761090996096 model_lib_v2.py:652] Step 201400 per-step time 0.240s loss=0.544
    INFO:tensorflow:Step 201500 per-step time 0.248s loss=0.494
    I1011 01:02:25.242851 139761090996096 model_lib_v2.py:652] Step 201500 per-step time 0.248s loss=0.494
    INFO:tensorflow:Step 201600 per-step time 0.257s loss=0.751
    I1011 01:02:50.165370 139761090996096 model_lib_v2.py:652] Step 201600 per-step time 0.257s loss=0.751
    INFO:tensorflow:Step 201700 per-step time 0.259s loss=0.891
    I1011 01:03:15.008761 139761090996096 model_lib_v2.py:652] Step 201700 per-step time 0.259s loss=0.891
    INFO:tensorflow:Step 201800 per-step time 0.244s loss=0.810
    I1011 01:03:39.854956 139761090996096 model_lib_v2.py:652] Step 201800 per-step time 0.244s loss=0.810
    INFO:tensorflow:Step 201900 per-step time 0.250s loss=0.902
    I1011 01:04:05.000351 139761090996096 model_lib_v2.py:652] Step 201900 per-step time 0.250s loss=0.902
    INFO:tensorflow:Step 202000 per-step time 0.248s loss=0.682
    I1011 01:04:29.722260 139761090996096 model_lib_v2.py:652] Step 202000 per-step time 0.248s loss=0.682
    INFO:tensorflow:Step 202100 per-step time 0.243s loss=0.659
    I1011 01:04:55.636145 139761090996096 model_lib_v2.py:652] Step 202100 per-step time 0.243s loss=0.659
    INFO:tensorflow:Step 202200 per-step time 0.258s loss=0.768
    I1011 01:05:20.647971 139761090996096 model_lib_v2.py:652] Step 202200 per-step time 0.258s loss=0.768
    INFO:tensorflow:Step 202300 per-step time 0.266s loss=0.682
    I1011 01:05:45.562426 139761090996096 model_lib_v2.py:652] Step 202300 per-step time 0.266s loss=0.682
    INFO:tensorflow:Step 202400 per-step time 0.239s loss=0.603
    I1011 01:06:10.414262 139761090996096 model_lib_v2.py:652] Step 202400 per-step time 0.239s loss=0.603
    INFO:tensorflow:Step 202500 per-step time 0.242s loss=0.753
    I1011 01:06:35.292265 139761090996096 model_lib_v2.py:652] Step 202500 per-step time 0.242s loss=0.753
    INFO:tensorflow:Step 202600 per-step time 0.256s loss=0.830
    I1011 01:07:00.301616 139761090996096 model_lib_v2.py:652] Step 202600 per-step time 0.256s loss=0.830
    INFO:tensorflow:Step 202700 per-step time 0.258s loss=0.594
    I1011 01:07:25.273942 139761090996096 model_lib_v2.py:652] Step 202700 per-step time 0.258s loss=0.594
    INFO:tensorflow:Step 202800 per-step time 0.247s loss=0.334
    I1011 01:07:50.181856 139761090996096 model_lib_v2.py:652] Step 202800 per-step time 0.247s loss=0.334
    INFO:tensorflow:Step 202900 per-step time 0.248s loss=0.601
    I1011 01:08:14.956193 139761090996096 model_lib_v2.py:652] Step 202900 per-step time 0.248s loss=0.601
    INFO:tensorflow:Step 203000 per-step time 0.255s loss=0.859
    I1011 01:08:39.830144 139761090996096 model_lib_v2.py:652] Step 203000 per-step time 0.255s loss=0.859
    INFO:tensorflow:Step 203100 per-step time 0.253s loss=0.554
    I1011 01:09:06.109963 139761090996096 model_lib_v2.py:652] Step 203100 per-step time 0.253s loss=0.554
    INFO:tensorflow:Step 203200 per-step time 0.249s loss=0.718
    I1011 01:09:31.201462 139761090996096 model_lib_v2.py:652] Step 203200 per-step time 0.249s loss=0.718
    INFO:tensorflow:Step 203300 per-step time 0.244s loss=0.532
    I1011 01:09:56.088010 139761090996096 model_lib_v2.py:652] Step 203300 per-step time 0.244s loss=0.532
    INFO:tensorflow:Step 203400 per-step time 0.239s loss=0.708
    I1011 01:10:20.923770 139761090996096 model_lib_v2.py:652] Step 203400 per-step time 0.239s loss=0.708
    INFO:tensorflow:Step 203500 per-step time 0.255s loss=0.780
    I1011 01:10:45.956094 139761090996096 model_lib_v2.py:652] Step 203500 per-step time 0.255s loss=0.780
    INFO:tensorflow:Step 203600 per-step time 0.247s loss=0.667
    I1011 01:11:10.856154 139761090996096 model_lib_v2.py:652] Step 203600 per-step time 0.247s loss=0.667
    INFO:tensorflow:Step 203700 per-step time 0.258s loss=0.534
    I1011 01:11:35.724965 139761090996096 model_lib_v2.py:652] Step 203700 per-step time 0.258s loss=0.534
    INFO:tensorflow:Step 203800 per-step time 0.235s loss=0.895
    I1011 01:12:00.602851 139761090996096 model_lib_v2.py:652] Step 203800 per-step time 0.235s loss=0.895
    INFO:tensorflow:Step 203900 per-step time 0.239s loss=0.873
    I1011 01:12:25.528255 139761090996096 model_lib_v2.py:652] Step 203900 per-step time 0.239s loss=0.873
    INFO:tensorflow:Step 204000 per-step time 0.242s loss=0.566
    I1011 01:12:50.572595 139761090996096 model_lib_v2.py:652] Step 204000 per-step time 0.242s loss=0.566
    INFO:tensorflow:Step 204100 per-step time 0.249s loss=0.620
    I1011 01:13:16.408179 139761090996096 model_lib_v2.py:652] Step 204100 per-step time 0.249s loss=0.620
    INFO:tensorflow:Step 204200 per-step time 0.247s loss=0.730
    I1011 01:13:41.385190 139761090996096 model_lib_v2.py:652] Step 204200 per-step time 0.247s loss=0.730
    INFO:tensorflow:Step 204300 per-step time 0.251s loss=0.542
    I1011 01:14:06.300069 139761090996096 model_lib_v2.py:652] Step 204300 per-step time 0.251s loss=0.542
    INFO:tensorflow:Step 204400 per-step time 0.255s loss=0.645
    I1011 01:14:31.467651 139761090996096 model_lib_v2.py:652] Step 204400 per-step time 0.255s loss=0.645
    INFO:tensorflow:Step 204500 per-step time 0.239s loss=0.696
    I1011 01:14:56.420399 139761090996096 model_lib_v2.py:652] Step 204500 per-step time 0.239s loss=0.696
    INFO:tensorflow:Step 204600 per-step time 0.248s loss=0.652
    I1011 01:15:21.227864 139761090996096 model_lib_v2.py:652] Step 204600 per-step time 0.248s loss=0.652
    INFO:tensorflow:Step 204700 per-step time 0.241s loss=0.645
    I1011 01:15:46.276446 139761090996096 model_lib_v2.py:652] Step 204700 per-step time 0.241s loss=0.645
    INFO:tensorflow:Step 204800 per-step time 0.238s loss=0.712
    I1011 01:16:11.253310 139761090996096 model_lib_v2.py:652] Step 204800 per-step time 0.238s loss=0.712
    INFO:tensorflow:Step 204900 per-step time 0.242s loss=0.672
    I1011 01:16:36.271347 139761090996096 model_lib_v2.py:652] Step 204900 per-step time 0.242s loss=0.672
    INFO:tensorflow:Step 205000 per-step time 0.253s loss=0.801
    I1011 01:17:01.221201 139761090996096 model_lib_v2.py:652] Step 205000 per-step time 0.253s loss=0.801
    INFO:tensorflow:Step 205100 per-step time 0.237s loss=0.634
    I1011 01:17:26.976152 139761090996096 model_lib_v2.py:652] Step 205100 per-step time 0.237s loss=0.634
    INFO:tensorflow:Step 205200 per-step time 0.251s loss=0.636
    I1011 01:17:51.921261 139761090996096 model_lib_v2.py:652] Step 205200 per-step time 0.251s loss=0.636
    INFO:tensorflow:Step 205300 per-step time 0.250s loss=0.539
    I1011 01:18:16.791806 139761090996096 model_lib_v2.py:652] Step 205300 per-step time 0.250s loss=0.539
    INFO:tensorflow:Step 205400 per-step time 0.250s loss=0.507
    I1011 01:18:41.976065 139761090996096 model_lib_v2.py:652] Step 205400 per-step time 0.250s loss=0.507
    INFO:tensorflow:Step 205500 per-step time 0.245s loss=0.815
    I1011 01:19:06.966115 139761090996096 model_lib_v2.py:652] Step 205500 per-step time 0.245s loss=0.815
    INFO:tensorflow:Step 205600 per-step time 0.252s loss=0.644
    I1011 01:19:32.055863 139761090996096 model_lib_v2.py:652] Step 205600 per-step time 0.252s loss=0.644
    INFO:tensorflow:Step 205700 per-step time 0.248s loss=0.727
    I1011 01:19:57.102828 139761090996096 model_lib_v2.py:652] Step 205700 per-step time 0.248s loss=0.727
    INFO:tensorflow:Step 205800 per-step time 0.246s loss=0.750
    I1011 01:20:21.929219 139761090996096 model_lib_v2.py:652] Step 205800 per-step time 0.246s loss=0.750
    INFO:tensorflow:Step 205900 per-step time 0.249s loss=0.568
    I1011 01:20:46.872021 139761090996096 model_lib_v2.py:652] Step 205900 per-step time 0.249s loss=0.568
    INFO:tensorflow:Step 206000 per-step time 0.248s loss=0.731
    I1011 01:21:11.826402 139761090996096 model_lib_v2.py:652] Step 206000 per-step time 0.248s loss=0.731
    INFO:tensorflow:Step 206100 per-step time 0.265s loss=0.562
    I1011 01:21:37.736643 139761090996096 model_lib_v2.py:652] Step 206100 per-step time 0.265s loss=0.562
    INFO:tensorflow:Step 206200 per-step time 0.251s loss=0.502
    I1011 01:22:02.583608 139761090996096 model_lib_v2.py:652] Step 206200 per-step time 0.251s loss=0.502
    INFO:tensorflow:Step 206300 per-step time 0.246s loss=0.715
    I1011 01:22:27.299751 139761090996096 model_lib_v2.py:652] Step 206300 per-step time 0.246s loss=0.715
    INFO:tensorflow:Step 206400 per-step time 0.252s loss=0.676
    I1011 01:22:52.227036 139761090996096 model_lib_v2.py:652] Step 206400 per-step time 0.252s loss=0.676
    INFO:tensorflow:Step 206500 per-step time 0.248s loss=0.327
    I1011 01:23:17.168555 139761090996096 model_lib_v2.py:652] Step 206500 per-step time 0.248s loss=0.327
    INFO:tensorflow:Step 206600 per-step time 0.256s loss=0.505
    I1011 01:23:42.107990 139761090996096 model_lib_v2.py:652] Step 206600 per-step time 0.256s loss=0.505
    INFO:tensorflow:Step 206700 per-step time 0.241s loss=0.588
    I1011 01:24:06.948675 139761090996096 model_lib_v2.py:652] Step 206700 per-step time 0.241s loss=0.588
    INFO:tensorflow:Step 206800 per-step time 0.250s loss=0.744
    I1011 01:24:32.063124 139761090996096 model_lib_v2.py:652] Step 206800 per-step time 0.250s loss=0.744
    INFO:tensorflow:Step 206900 per-step time 0.254s loss=0.829
    I1011 01:24:57.241128 139761090996096 model_lib_v2.py:652] Step 206900 per-step time 0.254s loss=0.829
    INFO:tensorflow:Step 207000 per-step time 0.242s loss=0.785
    I1011 01:25:22.177276 139761090996096 model_lib_v2.py:652] Step 207000 per-step time 0.242s loss=0.785
    INFO:tensorflow:Step 207100 per-step time 0.255s loss=0.634
    I1011 01:25:47.952721 139761090996096 model_lib_v2.py:652] Step 207100 per-step time 0.255s loss=0.634
    INFO:tensorflow:Step 207200 per-step time 0.251s loss=0.513
    I1011 01:26:12.924876 139761090996096 model_lib_v2.py:652] Step 207200 per-step time 0.251s loss=0.513
    INFO:tensorflow:Step 207300 per-step time 0.260s loss=0.709
    I1011 01:26:37.879603 139761090996096 model_lib_v2.py:652] Step 207300 per-step time 0.260s loss=0.709
    INFO:tensorflow:Step 207400 per-step time 0.261s loss=0.577
    I1011 01:27:02.877752 139761090996096 model_lib_v2.py:652] Step 207400 per-step time 0.261s loss=0.577
    INFO:tensorflow:Step 207500 per-step time 0.243s loss=1.075
    I1011 01:27:27.815654 139761090996096 model_lib_v2.py:652] Step 207500 per-step time 0.243s loss=1.075
    INFO:tensorflow:Step 207600 per-step time 0.241s loss=0.657
    I1011 01:27:52.705189 139761090996096 model_lib_v2.py:652] Step 207600 per-step time 0.241s loss=0.657
    INFO:tensorflow:Step 207700 per-step time 0.238s loss=0.743
    I1011 01:28:17.601536 139761090996096 model_lib_v2.py:652] Step 207700 per-step time 0.238s loss=0.743
    INFO:tensorflow:Step 207800 per-step time 0.251s loss=0.651
    I1011 01:28:42.500970 139761090996096 model_lib_v2.py:652] Step 207800 per-step time 0.251s loss=0.651
    INFO:tensorflow:Step 207900 per-step time 0.241s loss=0.723
    I1011 01:29:07.515565 139761090996096 model_lib_v2.py:652] Step 207900 per-step time 0.241s loss=0.723
    INFO:tensorflow:Step 208000 per-step time 0.242s loss=0.717
    I1011 01:29:32.353377 139761090996096 model_lib_v2.py:652] Step 208000 per-step time 0.242s loss=0.717
    INFO:tensorflow:Step 208100 per-step time 0.248s loss=0.676
    I1011 01:29:58.369621 139761090996096 model_lib_v2.py:652] Step 208100 per-step time 0.248s loss=0.676
    INFO:tensorflow:Step 208200 per-step time 0.261s loss=0.604
    I1011 01:30:23.343600 139761090996096 model_lib_v2.py:652] Step 208200 per-step time 0.261s loss=0.604
    INFO:tensorflow:Step 208300 per-step time 0.259s loss=0.552
    I1011 01:30:48.188107 139761090996096 model_lib_v2.py:652] Step 208300 per-step time 0.259s loss=0.552
    INFO:tensorflow:Step 208400 per-step time 0.251s loss=0.822
    I1011 01:31:13.133085 139761090996096 model_lib_v2.py:652] Step 208400 per-step time 0.251s loss=0.822
    INFO:tensorflow:Step 208500 per-step time 0.253s loss=0.877
    I1011 01:31:38.101141 139761090996096 model_lib_v2.py:652] Step 208500 per-step time 0.253s loss=0.877
    INFO:tensorflow:Step 208600 per-step time 0.255s loss=0.607
    I1011 01:32:02.999354 139761090996096 model_lib_v2.py:652] Step 208600 per-step time 0.255s loss=0.607
    INFO:tensorflow:Step 208700 per-step time 0.254s loss=0.593
    I1011 01:32:27.787756 139761090996096 model_lib_v2.py:652] Step 208700 per-step time 0.254s loss=0.593
    INFO:tensorflow:Step 208800 per-step time 0.256s loss=0.715
    I1011 01:32:52.623644 139761090996096 model_lib_v2.py:652] Step 208800 per-step time 0.256s loss=0.715
    INFO:tensorflow:Step 208900 per-step time 0.253s loss=0.752
    I1011 01:33:17.549691 139761090996096 model_lib_v2.py:652] Step 208900 per-step time 0.253s loss=0.752
    INFO:tensorflow:Step 209000 per-step time 0.252s loss=0.917
    I1011 01:33:42.530963 139761090996096 model_lib_v2.py:652] Step 209000 per-step time 0.252s loss=0.917
    INFO:tensorflow:Step 209100 per-step time 0.267s loss=0.437
    I1011 01:34:08.298487 139761090996096 model_lib_v2.py:652] Step 209100 per-step time 0.267s loss=0.437
    INFO:tensorflow:Step 209200 per-step time 0.255s loss=0.608
    I1011 01:34:33.170801 139761090996096 model_lib_v2.py:652] Step 209200 per-step time 0.255s loss=0.608
    INFO:tensorflow:Step 209300 per-step time 0.257s loss=0.549
    I1011 01:34:58.120947 139761090996096 model_lib_v2.py:652] Step 209300 per-step time 0.257s loss=0.549
    INFO:tensorflow:Step 209400 per-step time 0.242s loss=0.574
    I1011 01:35:23.246900 139761090996096 model_lib_v2.py:652] Step 209400 per-step time 0.242s loss=0.574
    INFO:tensorflow:Step 209500 per-step time 0.257s loss=0.676
    I1011 01:35:48.124858 139761090996096 model_lib_v2.py:652] Step 209500 per-step time 0.257s loss=0.676
    INFO:tensorflow:Step 209600 per-step time 0.258s loss=0.685
    I1011 01:36:12.829804 139761090996096 model_lib_v2.py:652] Step 209600 per-step time 0.258s loss=0.685
    INFO:tensorflow:Step 209700 per-step time 0.237s loss=0.612
    I1011 01:36:37.796067 139761090996096 model_lib_v2.py:652] Step 209700 per-step time 0.237s loss=0.612
    INFO:tensorflow:Step 209800 per-step time 0.246s loss=0.617
    I1011 01:37:02.556217 139761090996096 model_lib_v2.py:652] Step 209800 per-step time 0.246s loss=0.617
    INFO:tensorflow:Step 209900 per-step time 0.255s loss=0.832
    I1011 01:37:27.465626 139761090996096 model_lib_v2.py:652] Step 209900 per-step time 0.255s loss=0.832
    INFO:tensorflow:Step 210000 per-step time 0.239s loss=0.435
    I1011 01:37:52.285662 139761090996096 model_lib_v2.py:652] Step 210000 per-step time 0.239s loss=0.435
    INFO:tensorflow:Step 210100 per-step time 0.242s loss=0.544
    I1011 01:38:17.995741 139761090996096 model_lib_v2.py:652] Step 210100 per-step time 0.242s loss=0.544
    INFO:tensorflow:Step 210200 per-step time 0.255s loss=0.865
    I1011 01:38:42.879224 139761090996096 model_lib_v2.py:652] Step 210200 per-step time 0.255s loss=0.865
    INFO:tensorflow:Step 210300 per-step time 0.271s loss=0.401
    I1011 01:39:07.738950 139761090996096 model_lib_v2.py:652] Step 210300 per-step time 0.271s loss=0.401
    INFO:tensorflow:Step 210400 per-step time 0.242s loss=0.868
    I1011 01:39:32.505460 139761090996096 model_lib_v2.py:652] Step 210400 per-step time 0.242s loss=0.868
    INFO:tensorflow:Step 210500 per-step time 0.259s loss=0.764
    I1011 01:39:57.298598 139761090996096 model_lib_v2.py:652] Step 210500 per-step time 0.259s loss=0.764
    INFO:tensorflow:Step 210600 per-step time 0.245s loss=0.768
    I1011 01:40:22.397796 139761090996096 model_lib_v2.py:652] Step 210600 per-step time 0.245s loss=0.768
    INFO:tensorflow:Step 210700 per-step time 0.260s loss=0.883
    I1011 01:40:47.226496 139761090996096 model_lib_v2.py:652] Step 210700 per-step time 0.260s loss=0.883
    INFO:tensorflow:Step 210800 per-step time 0.261s loss=0.837
    I1011 01:41:12.051688 139761090996096 model_lib_v2.py:652] Step 210800 per-step time 0.261s loss=0.837
    INFO:tensorflow:Step 210900 per-step time 0.248s loss=0.461
    I1011 01:41:37.004937 139761090996096 model_lib_v2.py:652] Step 210900 per-step time 0.248s loss=0.461
    INFO:tensorflow:Step 211000 per-step time 0.247s loss=0.888
    I1011 01:42:01.902523 139761090996096 model_lib_v2.py:652] Step 211000 per-step time 0.247s loss=0.888
    INFO:tensorflow:Step 211100 per-step time 0.244s loss=0.571
    I1011 01:42:27.488337 139761090996096 model_lib_v2.py:652] Step 211100 per-step time 0.244s loss=0.571
    INFO:tensorflow:Step 211200 per-step time 0.253s loss=0.780
    I1011 01:42:52.430458 139761090996096 model_lib_v2.py:652] Step 211200 per-step time 0.253s loss=0.780
    INFO:tensorflow:Step 211300 per-step time 0.241s loss=0.674
    I1011 01:43:17.137506 139761090996096 model_lib_v2.py:652] Step 211300 per-step time 0.241s loss=0.674
    INFO:tensorflow:Step 211400 per-step time 0.257s loss=0.751
    I1011 01:43:41.829149 139761090996096 model_lib_v2.py:652] Step 211400 per-step time 0.257s loss=0.751
    INFO:tensorflow:Step 211500 per-step time 0.234s loss=0.862
    I1011 01:44:06.755947 139761090996096 model_lib_v2.py:652] Step 211500 per-step time 0.234s loss=0.862
    INFO:tensorflow:Step 211600 per-step time 0.250s loss=0.833
    I1011 01:44:31.658987 139761090996096 model_lib_v2.py:652] Step 211600 per-step time 0.250s loss=0.833
    INFO:tensorflow:Step 211700 per-step time 0.245s loss=0.778
    I1011 01:44:56.533020 139761090996096 model_lib_v2.py:652] Step 211700 per-step time 0.245s loss=0.778
    INFO:tensorflow:Step 211800 per-step time 0.245s loss=0.967
    I1011 01:45:21.487072 139761090996096 model_lib_v2.py:652] Step 211800 per-step time 0.245s loss=0.967
    INFO:tensorflow:Step 211900 per-step time 0.250s loss=0.743
    I1011 01:45:46.482584 139761090996096 model_lib_v2.py:652] Step 211900 per-step time 0.250s loss=0.743
    INFO:tensorflow:Step 212000 per-step time 0.247s loss=0.803
    I1011 01:46:11.200087 139761090996096 model_lib_v2.py:652] Step 212000 per-step time 0.247s loss=0.803
    INFO:tensorflow:Step 212100 per-step time 0.255s loss=0.505
    I1011 01:46:36.864535 139761090996096 model_lib_v2.py:652] Step 212100 per-step time 0.255s loss=0.505
    INFO:tensorflow:Step 212200 per-step time 0.256s loss=0.583
    I1011 01:47:01.678955 139761090996096 model_lib_v2.py:652] Step 212200 per-step time 0.256s loss=0.583
    INFO:tensorflow:Step 212300 per-step time 0.253s loss=0.750
    I1011 01:47:26.518431 139761090996096 model_lib_v2.py:652] Step 212300 per-step time 0.253s loss=0.750
    INFO:tensorflow:Step 212400 per-step time 0.254s loss=0.906
    I1011 01:47:51.318953 139761090996096 model_lib_v2.py:652] Step 212400 per-step time 0.254s loss=0.906
    INFO:tensorflow:Step 212500 per-step time 0.239s loss=0.772
    I1011 01:48:15.971212 139761090996096 model_lib_v2.py:652] Step 212500 per-step time 0.239s loss=0.772
    INFO:tensorflow:Step 212600 per-step time 0.235s loss=0.669
    I1011 01:48:40.678235 139761090996096 model_lib_v2.py:652] Step 212600 per-step time 0.235s loss=0.669
    INFO:tensorflow:Step 212700 per-step time 0.243s loss=0.525
    I1011 01:49:05.479649 139761090996096 model_lib_v2.py:652] Step 212700 per-step time 0.243s loss=0.525
    INFO:tensorflow:Step 212800 per-step time 0.248s loss=0.586
    I1011 01:49:30.511288 139761090996096 model_lib_v2.py:652] Step 212800 per-step time 0.248s loss=0.586
    INFO:tensorflow:Step 212900 per-step time 0.252s loss=0.579
    I1011 01:49:55.366325 139761090996096 model_lib_v2.py:652] Step 212900 per-step time 0.252s loss=0.579
    INFO:tensorflow:Step 213000 per-step time 0.258s loss=0.873
    I1011 01:50:20.327723 139761090996096 model_lib_v2.py:652] Step 213000 per-step time 0.258s loss=0.873
    INFO:tensorflow:Step 213100 per-step time 0.261s loss=0.690
    I1011 01:50:46.324160 139761090996096 model_lib_v2.py:652] Step 213100 per-step time 0.261s loss=0.690
    INFO:tensorflow:Step 213200 per-step time 0.245s loss=0.399
    I1011 01:51:11.100947 139761090996096 model_lib_v2.py:652] Step 213200 per-step time 0.245s loss=0.399
    INFO:tensorflow:Step 213300 per-step time 0.246s loss=0.541
    I1011 01:51:36.082515 139761090996096 model_lib_v2.py:652] Step 213300 per-step time 0.246s loss=0.541
    INFO:tensorflow:Step 213400 per-step time 0.249s loss=0.652
    I1011 01:52:00.839400 139761090996096 model_lib_v2.py:652] Step 213400 per-step time 0.249s loss=0.652
    INFO:tensorflow:Step 213500 per-step time 0.263s loss=0.834
    I1011 01:52:25.571516 139761090996096 model_lib_v2.py:652] Step 213500 per-step time 0.263s loss=0.834
    INFO:tensorflow:Step 213600 per-step time 0.252s loss=0.664
    I1011 01:52:50.427602 139761090996096 model_lib_v2.py:652] Step 213600 per-step time 0.252s loss=0.664
    INFO:tensorflow:Step 213700 per-step time 0.251s loss=0.580
    I1011 01:53:15.180377 139761090996096 model_lib_v2.py:652] Step 213700 per-step time 0.251s loss=0.580
    INFO:tensorflow:Step 213800 per-step time 0.258s loss=0.517
    I1011 01:53:39.899057 139761090996096 model_lib_v2.py:652] Step 213800 per-step time 0.258s loss=0.517
    INFO:tensorflow:Step 213900 per-step time 0.245s loss=0.647
    I1011 01:54:04.675355 139761090996096 model_lib_v2.py:652] Step 213900 per-step time 0.245s loss=0.647
    INFO:tensorflow:Step 214000 per-step time 0.236s loss=0.717
    I1011 01:54:29.329398 139761090996096 model_lib_v2.py:652] Step 214000 per-step time 0.236s loss=0.717
    INFO:tensorflow:Step 214100 per-step time 0.258s loss=0.710
    I1011 01:54:55.117761 139761090996096 model_lib_v2.py:652] Step 214100 per-step time 0.258s loss=0.710
    INFO:tensorflow:Step 214200 per-step time 0.248s loss=0.609
    I1011 01:55:19.990766 139761090996096 model_lib_v2.py:652] Step 214200 per-step time 0.248s loss=0.609
    INFO:tensorflow:Step 214300 per-step time 0.247s loss=0.757
    I1011 01:55:44.970017 139761090996096 model_lib_v2.py:652] Step 214300 per-step time 0.247s loss=0.757
    INFO:tensorflow:Step 214400 per-step time 0.266s loss=0.496
    I1011 01:56:09.924934 139761090996096 model_lib_v2.py:652] Step 214400 per-step time 0.266s loss=0.496
    INFO:tensorflow:Step 214500 per-step time 0.256s loss=0.557
    I1011 01:56:34.614270 139761090996096 model_lib_v2.py:652] Step 214500 per-step time 0.256s loss=0.557
    INFO:tensorflow:Step 214600 per-step time 0.247s loss=0.562
    I1011 01:56:59.405960 139761090996096 model_lib_v2.py:652] Step 214600 per-step time 0.247s loss=0.562
    INFO:tensorflow:Step 214700 per-step time 0.258s loss=0.981
    I1011 01:57:24.267160 139761090996096 model_lib_v2.py:652] Step 214700 per-step time 0.258s loss=0.981
    INFO:tensorflow:Step 214800 per-step time 0.250s loss=0.430
    I1011 01:57:49.035430 139761090996096 model_lib_v2.py:652] Step 214800 per-step time 0.250s loss=0.430
    INFO:tensorflow:Step 214900 per-step time 0.252s loss=0.939
    I1011 01:58:13.872516 139761090996096 model_lib_v2.py:652] Step 214900 per-step time 0.252s loss=0.939
    INFO:tensorflow:Step 215000 per-step time 0.246s loss=0.533
    I1011 01:58:38.777545 139761090996096 model_lib_v2.py:652] Step 215000 per-step time 0.246s loss=0.533
    INFO:tensorflow:Step 215100 per-step time 0.245s loss=1.048
    I1011 01:59:04.514114 139761090996096 model_lib_v2.py:652] Step 215100 per-step time 0.245s loss=1.048
    INFO:tensorflow:Step 215200 per-step time 0.255s loss=0.556
    I1011 01:59:29.327873 139761090996096 model_lib_v2.py:652] Step 215200 per-step time 0.255s loss=0.556
    INFO:tensorflow:Step 215300 per-step time 0.242s loss=0.852
    I1011 01:59:54.044978 139761090996096 model_lib_v2.py:652] Step 215300 per-step time 0.242s loss=0.852
    INFO:tensorflow:Step 215400 per-step time 0.256s loss=1.009
    I1011 02:00:18.797403 139761090996096 model_lib_v2.py:652] Step 215400 per-step time 0.256s loss=1.009
    INFO:tensorflow:Step 215500 per-step time 0.232s loss=0.947
    I1011 02:00:43.406742 139761090996096 model_lib_v2.py:652] Step 215500 per-step time 0.232s loss=0.947
    INFO:tensorflow:Step 215600 per-step time 0.244s loss=0.600
    I1011 02:01:08.278771 139761090996096 model_lib_v2.py:652] Step 215600 per-step time 0.244s loss=0.600
    INFO:tensorflow:Step 215700 per-step time 0.247s loss=0.453
    I1011 02:01:33.011339 139761090996096 model_lib_v2.py:652] Step 215700 per-step time 0.247s loss=0.453
    INFO:tensorflow:Step 215800 per-step time 0.250s loss=0.624
    I1011 02:01:57.668747 139761090996096 model_lib_v2.py:652] Step 215800 per-step time 0.250s loss=0.624
    INFO:tensorflow:Step 215900 per-step time 0.243s loss=0.702
    I1011 02:02:22.403085 139761090996096 model_lib_v2.py:652] Step 215900 per-step time 0.243s loss=0.702
    INFO:tensorflow:Step 216000 per-step time 0.246s loss=0.475
    I1011 02:02:47.102882 139761090996096 model_lib_v2.py:652] Step 216000 per-step time 0.246s loss=0.475
    INFO:tensorflow:Step 216100 per-step time 0.250s loss=0.588
    I1011 02:03:12.516134 139761090996096 model_lib_v2.py:652] Step 216100 per-step time 0.250s loss=0.588
    INFO:tensorflow:Step 216200 per-step time 0.245s loss=0.781
    I1011 02:03:37.190771 139761090996096 model_lib_v2.py:652] Step 216200 per-step time 0.245s loss=0.781
    INFO:tensorflow:Step 216300 per-step time 0.239s loss=0.619
    I1011 02:04:01.783405 139761090996096 model_lib_v2.py:652] Step 216300 per-step time 0.239s loss=0.619
    INFO:tensorflow:Step 216400 per-step time 0.247s loss=0.577
    I1011 02:04:26.378849 139761090996096 model_lib_v2.py:652] Step 216400 per-step time 0.247s loss=0.577
    INFO:tensorflow:Step 216500 per-step time 0.257s loss=0.846
    I1011 02:04:51.147125 139761090996096 model_lib_v2.py:652] Step 216500 per-step time 0.257s loss=0.846
    INFO:tensorflow:Step 216600 per-step time 0.248s loss=0.563
    I1011 02:05:15.848548 139761090996096 model_lib_v2.py:652] Step 216600 per-step time 0.248s loss=0.563
    INFO:tensorflow:Step 216700 per-step time 0.252s loss=0.534
    I1011 02:05:40.651391 139761090996096 model_lib_v2.py:652] Step 216700 per-step time 0.252s loss=0.534
    INFO:tensorflow:Step 216800 per-step time 0.252s loss=0.653
    I1011 02:06:05.565305 139761090996096 model_lib_v2.py:652] Step 216800 per-step time 0.252s loss=0.653
    INFO:tensorflow:Step 216900 per-step time 0.251s loss=0.846
    I1011 02:06:30.721797 139761090996096 model_lib_v2.py:652] Step 216900 per-step time 0.251s loss=0.846
    INFO:tensorflow:Step 217000 per-step time 0.248s loss=0.515
    I1011 02:06:55.521783 139761090996096 model_lib_v2.py:652] Step 217000 per-step time 0.248s loss=0.515
    INFO:tensorflow:Step 217100 per-step time 0.257s loss=0.634
    I1011 02:07:21.274666 139761090996096 model_lib_v2.py:652] Step 217100 per-step time 0.257s loss=0.634
    INFO:tensorflow:Step 217200 per-step time 0.262s loss=0.453
    I1011 02:07:46.084781 139761090996096 model_lib_v2.py:652] Step 217200 per-step time 0.262s loss=0.453
    INFO:tensorflow:Step 217300 per-step time 0.246s loss=0.837
    I1011 02:08:10.832960 139761090996096 model_lib_v2.py:652] Step 217300 per-step time 0.246s loss=0.837
    INFO:tensorflow:Step 217400 per-step time 0.248s loss=0.588
    I1011 02:08:35.634956 139761090996096 model_lib_v2.py:652] Step 217400 per-step time 0.248s loss=0.588
    INFO:tensorflow:Step 217500 per-step time 0.239s loss=0.624
    I1011 02:09:00.301086 139761090996096 model_lib_v2.py:652] Step 217500 per-step time 0.239s loss=0.624
    INFO:tensorflow:Step 217600 per-step time 0.254s loss=0.480
    I1011 02:09:25.226658 139761090996096 model_lib_v2.py:652] Step 217600 per-step time 0.254s loss=0.480
    INFO:tensorflow:Step 217700 per-step time 0.243s loss=0.356
    I1011 02:09:50.053438 139761090996096 model_lib_v2.py:652] Step 217700 per-step time 0.243s loss=0.356
    INFO:tensorflow:Step 217800 per-step time 0.242s loss=0.684
    I1011 02:10:14.868359 139761090996096 model_lib_v2.py:652] Step 217800 per-step time 0.242s loss=0.684
    INFO:tensorflow:Step 217900 per-step time 0.256s loss=0.851
    I1011 02:10:39.682162 139761090996096 model_lib_v2.py:652] Step 217900 per-step time 0.256s loss=0.851
    INFO:tensorflow:Step 218000 per-step time 0.253s loss=1.085
    I1011 02:11:04.434250 139761090996096 model_lib_v2.py:652] Step 218000 per-step time 0.253s loss=1.085
    INFO:tensorflow:Step 218100 per-step time 0.253s loss=0.634
    I1011 02:11:30.456047 139761090996096 model_lib_v2.py:652] Step 218100 per-step time 0.253s loss=0.634
    INFO:tensorflow:Step 218200 per-step time 0.255s loss=0.703
    I1011 02:11:55.176907 139761090996096 model_lib_v2.py:652] Step 218200 per-step time 0.255s loss=0.703
    INFO:tensorflow:Step 218300 per-step time 0.252s loss=0.459
    I1011 02:12:19.921727 139761090996096 model_lib_v2.py:652] Step 218300 per-step time 0.252s loss=0.459
    INFO:tensorflow:Step 218400 per-step time 0.245s loss=0.625
    I1011 02:12:44.760458 139761090996096 model_lib_v2.py:652] Step 218400 per-step time 0.245s loss=0.625
    INFO:tensorflow:Step 218500 per-step time 0.253s loss=0.460
    I1011 02:13:09.438871 139761090996096 model_lib_v2.py:652] Step 218500 per-step time 0.253s loss=0.460
    INFO:tensorflow:Step 218600 per-step time 0.255s loss=0.573
    I1011 02:13:34.286453 139761090996096 model_lib_v2.py:652] Step 218600 per-step time 0.255s loss=0.573
    INFO:tensorflow:Step 218700 per-step time 0.253s loss=0.439
    I1011 02:13:59.194984 139761090996096 model_lib_v2.py:652] Step 218700 per-step time 0.253s loss=0.439
    INFO:tensorflow:Step 218800 per-step time 0.258s loss=0.552
    I1011 02:14:23.982257 139761090996096 model_lib_v2.py:652] Step 218800 per-step time 0.258s loss=0.552
    INFO:tensorflow:Step 218900 per-step time 0.249s loss=0.654
    I1011 02:14:48.975881 139761090996096 model_lib_v2.py:652] Step 218900 per-step time 0.249s loss=0.654
    INFO:tensorflow:Step 219000 per-step time 0.234s loss=0.559
    I1011 02:15:13.914769 139761090996096 model_lib_v2.py:652] Step 219000 per-step time 0.234s loss=0.559
    INFO:tensorflow:Step 219100 per-step time 0.266s loss=0.734
    I1011 02:15:39.679670 139761090996096 model_lib_v2.py:652] Step 219100 per-step time 0.266s loss=0.734
    INFO:tensorflow:Step 219200 per-step time 0.248s loss=1.008
    I1011 02:16:04.432662 139761090996096 model_lib_v2.py:652] Step 219200 per-step time 0.248s loss=1.008
    INFO:tensorflow:Step 219300 per-step time 0.262s loss=0.627
    I1011 02:16:29.286799 139761090996096 model_lib_v2.py:652] Step 219300 per-step time 0.262s loss=0.627
    INFO:tensorflow:Step 219400 per-step time 0.247s loss=1.020
    I1011 02:16:54.109627 139761090996096 model_lib_v2.py:652] Step 219400 per-step time 0.247s loss=1.020
    INFO:tensorflow:Step 219500 per-step time 0.254s loss=0.698
    I1011 02:17:18.916313 139761090996096 model_lib_v2.py:652] Step 219500 per-step time 0.254s loss=0.698
    INFO:tensorflow:Step 219600 per-step time 0.235s loss=0.587
    I1011 02:17:43.723211 139761090996096 model_lib_v2.py:652] Step 219600 per-step time 0.235s loss=0.587
    INFO:tensorflow:Step 219700 per-step time 0.244s loss=0.719
    I1011 02:18:08.651574 139761090996096 model_lib_v2.py:652] Step 219700 per-step time 0.244s loss=0.719
    INFO:tensorflow:Step 219800 per-step time 0.250s loss=0.648
    I1011 02:18:33.510368 139761090996096 model_lib_v2.py:652] Step 219800 per-step time 0.250s loss=0.648
    INFO:tensorflow:Step 219900 per-step time 0.249s loss=0.871
    I1011 02:18:58.405643 139761090996096 model_lib_v2.py:652] Step 219900 per-step time 0.249s loss=0.871
    INFO:tensorflow:Step 220000 per-step time 0.248s loss=0.851
    I1011 02:19:23.162464 139761090996096 model_lib_v2.py:652] Step 220000 per-step time 0.248s loss=0.851
    INFO:tensorflow:Step 220100 per-step time 0.244s loss=0.632
    I1011 02:19:48.828523 139761090996096 model_lib_v2.py:652] Step 220100 per-step time 0.244s loss=0.632
    INFO:tensorflow:Step 220200 per-step time 0.253s loss=0.623
    I1011 02:20:13.682018 139761090996096 model_lib_v2.py:652] Step 220200 per-step time 0.253s loss=0.623
    INFO:tensorflow:Step 220300 per-step time 0.242s loss=0.812
    I1011 02:20:38.582009 139761090996096 model_lib_v2.py:652] Step 220300 per-step time 0.242s loss=0.812
    INFO:tensorflow:Step 220400 per-step time 0.251s loss=0.751
    I1011 02:21:03.510887 139761090996096 model_lib_v2.py:652] Step 220400 per-step time 0.251s loss=0.751
    INFO:tensorflow:Step 220500 per-step time 0.246s loss=0.573
    I1011 02:21:28.284487 139761090996096 model_lib_v2.py:652] Step 220500 per-step time 0.246s loss=0.573
    INFO:tensorflow:Step 220600 per-step time 0.243s loss=0.703
    I1011 02:21:53.316401 139761090996096 model_lib_v2.py:652] Step 220600 per-step time 0.243s loss=0.703
    INFO:tensorflow:Step 220700 per-step time 0.238s loss=0.492
    I1011 02:22:18.205234 139761090996096 model_lib_v2.py:652] Step 220700 per-step time 0.238s loss=0.492
    INFO:tensorflow:Step 220800 per-step time 0.241s loss=0.559
    I1011 02:22:42.926253 139761090996096 model_lib_v2.py:652] Step 220800 per-step time 0.241s loss=0.559
    INFO:tensorflow:Step 220900 per-step time 0.247s loss=0.606
    I1011 02:23:07.818470 139761090996096 model_lib_v2.py:652] Step 220900 per-step time 0.247s loss=0.606
    INFO:tensorflow:Step 221000 per-step time 0.255s loss=0.410
    I1011 02:23:32.754854 139761090996096 model_lib_v2.py:652] Step 221000 per-step time 0.255s loss=0.410
    INFO:tensorflow:Step 221100 per-step time 0.237s loss=0.846
    I1011 02:23:58.479472 139761090996096 model_lib_v2.py:652] Step 221100 per-step time 0.237s loss=0.846
    INFO:tensorflow:Step 221200 per-step time 0.243s loss=0.591
    I1011 02:24:23.200482 139761090996096 model_lib_v2.py:652] Step 221200 per-step time 0.243s loss=0.591
    INFO:tensorflow:Step 221300 per-step time 0.259s loss=0.771
    I1011 02:24:48.080012 139761090996096 model_lib_v2.py:652] Step 221300 per-step time 0.259s loss=0.771
    INFO:tensorflow:Step 221400 per-step time 0.266s loss=0.607
    I1011 02:25:13.055659 139761090996096 model_lib_v2.py:652] Step 221400 per-step time 0.266s loss=0.607
    INFO:tensorflow:Step 221500 per-step time 0.233s loss=0.557
    I1011 02:25:37.886949 139761090996096 model_lib_v2.py:652] Step 221500 per-step time 0.233s loss=0.557
    INFO:tensorflow:Step 221600 per-step time 0.245s loss=0.641
    I1011 02:26:02.719291 139761090996096 model_lib_v2.py:652] Step 221600 per-step time 0.245s loss=0.641
    INFO:tensorflow:Step 221700 per-step time 0.252s loss=0.441
    I1011 02:26:27.574926 139761090996096 model_lib_v2.py:652] Step 221700 per-step time 0.252s loss=0.441
    INFO:tensorflow:Step 221800 per-step time 0.265s loss=0.587
    I1011 02:26:52.721192 139761090996096 model_lib_v2.py:652] Step 221800 per-step time 0.265s loss=0.587
    INFO:tensorflow:Step 221900 per-step time 0.237s loss=0.633
    I1011 02:27:17.752438 139761090996096 model_lib_v2.py:652] Step 221900 per-step time 0.237s loss=0.633
    INFO:tensorflow:Step 222000 per-step time 0.262s loss=0.795
    I1011 02:27:42.873104 139761090996096 model_lib_v2.py:652] Step 222000 per-step time 0.262s loss=0.795
    INFO:tensorflow:Step 222100 per-step time 0.261s loss=0.710
    I1011 02:28:08.670390 139761090996096 model_lib_v2.py:652] Step 222100 per-step time 0.261s loss=0.710
    INFO:tensorflow:Step 222200 per-step time 0.252s loss=0.651
    I1011 02:28:33.542855 139761090996096 model_lib_v2.py:652] Step 222200 per-step time 0.252s loss=0.651
    INFO:tensorflow:Step 222300 per-step time 0.241s loss=0.458
    I1011 02:28:58.318569 139761090996096 model_lib_v2.py:652] Step 222300 per-step time 0.241s loss=0.458
    INFO:tensorflow:Step 222400 per-step time 0.247s loss=0.568
    I1011 02:29:23.409828 139761090996096 model_lib_v2.py:652] Step 222400 per-step time 0.247s loss=0.568
    INFO:tensorflow:Step 222500 per-step time 0.249s loss=0.636
    I1011 02:29:48.145076 139761090996096 model_lib_v2.py:652] Step 222500 per-step time 0.249s loss=0.636
    INFO:tensorflow:Step 222600 per-step time 0.247s loss=0.789
    I1011 02:30:13.032829 139761090996096 model_lib_v2.py:652] Step 222600 per-step time 0.247s loss=0.789
    INFO:tensorflow:Step 222700 per-step time 0.242s loss=0.752
    I1011 02:30:37.886368 139761090996096 model_lib_v2.py:652] Step 222700 per-step time 0.242s loss=0.752
    INFO:tensorflow:Step 222800 per-step time 0.259s loss=0.473
    I1011 02:31:02.703771 139761090996096 model_lib_v2.py:652] Step 222800 per-step time 0.259s loss=0.473
    INFO:tensorflow:Step 222900 per-step time 0.254s loss=0.683
    I1011 02:31:27.605049 139761090996096 model_lib_v2.py:652] Step 222900 per-step time 0.254s loss=0.683
    INFO:tensorflow:Step 223000 per-step time 0.249s loss=0.727
    I1011 02:31:52.481820 139761090996096 model_lib_v2.py:652] Step 223000 per-step time 0.249s loss=0.727
    INFO:tensorflow:Step 223100 per-step time 0.242s loss=0.544
    I1011 02:32:18.545865 139761090996096 model_lib_v2.py:652] Step 223100 per-step time 0.242s loss=0.544
    INFO:tensorflow:Step 223200 per-step time 0.254s loss=0.841
    I1011 02:32:43.253335 139761090996096 model_lib_v2.py:652] Step 223200 per-step time 0.254s loss=0.841
    INFO:tensorflow:Step 223300 per-step time 0.247s loss=0.511
    I1011 02:33:08.083216 139761090996096 model_lib_v2.py:652] Step 223300 per-step time 0.247s loss=0.511
    INFO:tensorflow:Step 223400 per-step time 0.255s loss=0.285
    I1011 02:33:32.888849 139761090996096 model_lib_v2.py:652] Step 223400 per-step time 0.255s loss=0.285
    INFO:tensorflow:Step 223500 per-step time 0.254s loss=0.834
    I1011 02:33:57.778474 139761090996096 model_lib_v2.py:652] Step 223500 per-step time 0.254s loss=0.834
    INFO:tensorflow:Step 223600 per-step time 0.240s loss=0.675
    I1011 02:34:22.670392 139761090996096 model_lib_v2.py:652] Step 223600 per-step time 0.240s loss=0.675
    INFO:tensorflow:Step 223700 per-step time 0.249s loss=0.546
    I1011 02:34:47.513749 139761090996096 model_lib_v2.py:652] Step 223700 per-step time 0.249s loss=0.546
    INFO:tensorflow:Step 223800 per-step time 0.262s loss=0.602
    I1011 02:35:12.349110 139761090996096 model_lib_v2.py:652] Step 223800 per-step time 0.262s loss=0.602
    INFO:tensorflow:Step 223900 per-step time 0.247s loss=0.706
    I1011 02:35:37.267901 139761090996096 model_lib_v2.py:652] Step 223900 per-step time 0.247s loss=0.706
    INFO:tensorflow:Step 224000 per-step time 0.258s loss=0.805
    I1011 02:36:02.027897 139761090996096 model_lib_v2.py:652] Step 224000 per-step time 0.258s loss=0.805
    INFO:tensorflow:Step 224100 per-step time 0.248s loss=0.618
    I1011 02:36:27.883137 139761090996096 model_lib_v2.py:652] Step 224100 per-step time 0.248s loss=0.618
    INFO:tensorflow:Step 224200 per-step time 0.245s loss=0.589
    I1011 02:36:52.685918 139761090996096 model_lib_v2.py:652] Step 224200 per-step time 0.245s loss=0.589
    INFO:tensorflow:Step 224300 per-step time 0.260s loss=0.747
    I1011 02:37:17.722091 139761090996096 model_lib_v2.py:652] Step 224300 per-step time 0.260s loss=0.747
    INFO:tensorflow:Step 224400 per-step time 0.255s loss=0.596
    I1011 02:37:42.685792 139761090996096 model_lib_v2.py:652] Step 224400 per-step time 0.255s loss=0.596
    INFO:tensorflow:Step 224500 per-step time 0.247s loss=0.601
    I1011 02:38:07.523954 139761090996096 model_lib_v2.py:652] Step 224500 per-step time 0.247s loss=0.601
    INFO:tensorflow:Step 224600 per-step time 0.242s loss=0.583
    I1011 02:38:32.364105 139761090996096 model_lib_v2.py:652] Step 224600 per-step time 0.242s loss=0.583
    INFO:tensorflow:Step 224700 per-step time 0.248s loss=0.485
    I1011 02:38:57.249391 139761090996096 model_lib_v2.py:652] Step 224700 per-step time 0.248s loss=0.485
    INFO:tensorflow:Step 224800 per-step time 0.254s loss=0.701
    I1011 02:39:22.151798 139761090996096 model_lib_v2.py:652] Step 224800 per-step time 0.254s loss=0.701
    INFO:tensorflow:Step 224900 per-step time 0.265s loss=0.707
    I1011 02:39:46.983804 139761090996096 model_lib_v2.py:652] Step 224900 per-step time 0.265s loss=0.707
    INFO:tensorflow:Step 225000 per-step time 0.247s loss=0.419
    I1011 02:40:11.936630 139761090996096 model_lib_v2.py:652] Step 225000 per-step time 0.247s loss=0.419
    INFO:tensorflow:Step 225100 per-step time 0.253s loss=0.668
    I1011 02:40:37.765834 139761090996096 model_lib_v2.py:652] Step 225100 per-step time 0.253s loss=0.668
    INFO:tensorflow:Step 225200 per-step time 0.247s loss=0.638
    I1011 02:41:02.631078 139761090996096 model_lib_v2.py:652] Step 225200 per-step time 0.247s loss=0.638
    INFO:tensorflow:Step 225300 per-step time 0.238s loss=0.628
    I1011 02:41:27.521687 139761090996096 model_lib_v2.py:652] Step 225300 per-step time 0.238s loss=0.628
    INFO:tensorflow:Step 225400 per-step time 0.239s loss=0.660
    I1011 02:41:52.257140 139761090996096 model_lib_v2.py:652] Step 225400 per-step time 0.239s loss=0.660
    INFO:tensorflow:Step 225500 per-step time 0.250s loss=0.786
    I1011 02:42:17.117540 139761090996096 model_lib_v2.py:652] Step 225500 per-step time 0.250s loss=0.786
    INFO:tensorflow:Step 225600 per-step time 0.258s loss=0.782
    I1011 02:42:42.158755 139761090996096 model_lib_v2.py:652] Step 225600 per-step time 0.258s loss=0.782
    INFO:tensorflow:Step 225700 per-step time 0.245s loss=0.589
    I1011 02:43:07.089238 139761090996096 model_lib_v2.py:652] Step 225700 per-step time 0.245s loss=0.589
    INFO:tensorflow:Step 225800 per-step time 0.244s loss=0.713
    I1011 02:43:31.982564 139761090996096 model_lib_v2.py:652] Step 225800 per-step time 0.244s loss=0.713
    INFO:tensorflow:Step 225900 per-step time 0.257s loss=0.752
    I1011 02:43:56.964803 139761090996096 model_lib_v2.py:652] Step 225900 per-step time 0.257s loss=0.752
    INFO:tensorflow:Step 226000 per-step time 0.260s loss=0.477
    I1011 02:44:21.810895 139761090996096 model_lib_v2.py:652] Step 226000 per-step time 0.260s loss=0.477
    INFO:tensorflow:Step 226100 per-step time 0.260s loss=0.787
    I1011 02:44:47.797230 139761090996096 model_lib_v2.py:652] Step 226100 per-step time 0.260s loss=0.787
    INFO:tensorflow:Step 226200 per-step time 0.239s loss=0.637
    I1011 02:45:12.798587 139761090996096 model_lib_v2.py:652] Step 226200 per-step time 0.239s loss=0.637
    INFO:tensorflow:Step 226300 per-step time 0.255s loss=0.747
    I1011 02:45:37.766059 139761090996096 model_lib_v2.py:652] Step 226300 per-step time 0.255s loss=0.747
    INFO:tensorflow:Step 226400 per-step time 0.249s loss=0.727
    I1011 02:46:02.739769 139761090996096 model_lib_v2.py:652] Step 226400 per-step time 0.249s loss=0.727
    INFO:tensorflow:Step 226500 per-step time 0.253s loss=0.700
    I1011 02:46:27.672620 139761090996096 model_lib_v2.py:652] Step 226500 per-step time 0.253s loss=0.700
    INFO:tensorflow:Step 226600 per-step time 0.242s loss=0.774
    I1011 02:46:52.500691 139761090996096 model_lib_v2.py:652] Step 226600 per-step time 0.242s loss=0.774
    INFO:tensorflow:Step 226700 per-step time 0.251s loss=0.643
    I1011 02:47:17.383538 139761090996096 model_lib_v2.py:652] Step 226700 per-step time 0.251s loss=0.643
    INFO:tensorflow:Step 226800 per-step time 0.248s loss=0.565
    I1011 02:47:42.448069 139761090996096 model_lib_v2.py:652] Step 226800 per-step time 0.248s loss=0.565
    INFO:tensorflow:Step 226900 per-step time 0.242s loss=0.748
    I1011 02:48:07.350431 139761090996096 model_lib_v2.py:652] Step 226900 per-step time 0.242s loss=0.748
    INFO:tensorflow:Step 227000 per-step time 0.239s loss=0.645
    I1011 02:48:32.199634 139761090996096 model_lib_v2.py:652] Step 227000 per-step time 0.239s loss=0.645
    INFO:tensorflow:Step 227100 per-step time 0.241s loss=0.558
    I1011 02:48:57.977170 139761090996096 model_lib_v2.py:652] Step 227100 per-step time 0.241s loss=0.558
    INFO:tensorflow:Step 227200 per-step time 0.250s loss=0.831
    I1011 02:49:22.921557 139761090996096 model_lib_v2.py:652] Step 227200 per-step time 0.250s loss=0.831
    INFO:tensorflow:Step 227300 per-step time 0.246s loss=1.089
    I1011 02:49:47.789580 139761090996096 model_lib_v2.py:652] Step 227300 per-step time 0.246s loss=1.089
    INFO:tensorflow:Step 227400 per-step time 0.254s loss=0.608
    I1011 02:50:12.690046 139761090996096 model_lib_v2.py:652] Step 227400 per-step time 0.254s loss=0.608
    INFO:tensorflow:Step 227500 per-step time 0.255s loss=0.788
    I1011 02:50:37.536423 139761090996096 model_lib_v2.py:652] Step 227500 per-step time 0.255s loss=0.788
    INFO:tensorflow:Step 227600 per-step time 0.254s loss=0.596
    I1011 02:51:02.612342 139761090996096 model_lib_v2.py:652] Step 227600 per-step time 0.254s loss=0.596
    INFO:tensorflow:Step 227700 per-step time 0.254s loss=0.804
    I1011 02:51:27.497299 139761090996096 model_lib_v2.py:652] Step 227700 per-step time 0.254s loss=0.804
    INFO:tensorflow:Step 227800 per-step time 0.251s loss=0.892
    I1011 02:51:52.453567 139761090996096 model_lib_v2.py:652] Step 227800 per-step time 0.251s loss=0.892
    INFO:tensorflow:Step 227900 per-step time 0.241s loss=0.754
    I1011 02:52:17.315790 139761090996096 model_lib_v2.py:652] Step 227900 per-step time 0.241s loss=0.754
    INFO:tensorflow:Step 228000 per-step time 0.252s loss=0.634
    I1011 02:52:42.257342 139761090996096 model_lib_v2.py:652] Step 228000 per-step time 0.252s loss=0.634
    INFO:tensorflow:Step 228100 per-step time 0.246s loss=0.550
    I1011 02:53:08.295125 139761090996096 model_lib_v2.py:652] Step 228100 per-step time 0.246s loss=0.550
    INFO:tensorflow:Step 228200 per-step time 0.242s loss=0.551
    I1011 02:53:33.392243 139761090996096 model_lib_v2.py:652] Step 228200 per-step time 0.242s loss=0.551
    INFO:tensorflow:Step 228300 per-step time 0.247s loss=0.671
    I1011 02:53:58.293726 139761090996096 model_lib_v2.py:652] Step 228300 per-step time 0.247s loss=0.671
    INFO:tensorflow:Step 228400 per-step time 0.261s loss=0.765
    I1011 02:54:23.267839 139761090996096 model_lib_v2.py:652] Step 228400 per-step time 0.261s loss=0.765
    INFO:tensorflow:Step 228500 per-step time 0.262s loss=0.634
    I1011 02:54:48.217798 139761090996096 model_lib_v2.py:652] Step 228500 per-step time 0.262s loss=0.634
    INFO:tensorflow:Step 228600 per-step time 0.254s loss=0.691
    I1011 02:55:13.083078 139761090996096 model_lib_v2.py:652] Step 228600 per-step time 0.254s loss=0.691
    INFO:tensorflow:Step 228700 per-step time 0.241s loss=0.550
    I1011 02:55:37.981201 139761090996096 model_lib_v2.py:652] Step 228700 per-step time 0.241s loss=0.550
    INFO:tensorflow:Step 228800 per-step time 0.245s loss=0.748
    I1011 02:56:02.767979 139761090996096 model_lib_v2.py:652] Step 228800 per-step time 0.245s loss=0.748
    INFO:tensorflow:Step 228900 per-step time 0.249s loss=0.851
    I1011 02:56:27.692577 139761090996096 model_lib_v2.py:652] Step 228900 per-step time 0.249s loss=0.851
    INFO:tensorflow:Step 229000 per-step time 0.249s loss=0.599
    I1011 02:56:52.679783 139761090996096 model_lib_v2.py:652] Step 229000 per-step time 0.249s loss=0.599
    INFO:tensorflow:Step 229100 per-step time 0.252s loss=0.786
    I1011 02:57:18.597781 139761090996096 model_lib_v2.py:652] Step 229100 per-step time 0.252s loss=0.786
    INFO:tensorflow:Step 229200 per-step time 0.261s loss=0.697
    I1011 02:57:43.620475 139761090996096 model_lib_v2.py:652] Step 229200 per-step time 0.261s loss=0.697
    INFO:tensorflow:Step 229300 per-step time 0.253s loss=0.549
    I1011 02:58:08.497198 139761090996096 model_lib_v2.py:652] Step 229300 per-step time 0.253s loss=0.549
    INFO:tensorflow:Step 229400 per-step time 0.249s loss=0.719
    I1011 02:58:33.235990 139761090996096 model_lib_v2.py:652] Step 229400 per-step time 0.249s loss=0.719
    INFO:tensorflow:Step 229500 per-step time 0.263s loss=0.796
    I1011 02:58:58.094454 139761090996096 model_lib_v2.py:652] Step 229500 per-step time 0.263s loss=0.796
    INFO:tensorflow:Step 229600 per-step time 0.250s loss=0.445
    I1011 02:59:22.968461 139761090996096 model_lib_v2.py:652] Step 229600 per-step time 0.250s loss=0.445
    INFO:tensorflow:Step 229700 per-step time 0.246s loss=0.871
    I1011 02:59:47.910629 139761090996096 model_lib_v2.py:652] Step 229700 per-step time 0.246s loss=0.871
    INFO:tensorflow:Step 229800 per-step time 0.244s loss=0.624
    I1011 03:00:12.784734 139761090996096 model_lib_v2.py:652] Step 229800 per-step time 0.244s loss=0.624
    INFO:tensorflow:Step 229900 per-step time 0.244s loss=0.518
    I1011 03:00:37.612590 139761090996096 model_lib_v2.py:652] Step 229900 per-step time 0.244s loss=0.518
    INFO:tensorflow:Step 230000 per-step time 0.249s loss=0.530
    I1011 03:01:02.566595 139761090996096 model_lib_v2.py:652] Step 230000 per-step time 0.249s loss=0.530
    INFO:tensorflow:Step 230100 per-step time 0.254s loss=0.651
    I1011 03:01:28.313203 139761090996096 model_lib_v2.py:652] Step 230100 per-step time 0.254s loss=0.651
    INFO:tensorflow:Step 230200 per-step time 0.246s loss=0.810
    I1011 03:01:53.257481 139761090996096 model_lib_v2.py:652] Step 230200 per-step time 0.246s loss=0.810
    INFO:tensorflow:Step 230300 per-step time 0.277s loss=0.727
    I1011 03:02:18.197701 139761090996096 model_lib_v2.py:652] Step 230300 per-step time 0.277s loss=0.727
    INFO:tensorflow:Step 230400 per-step time 0.236s loss=0.815
    I1011 03:02:43.004497 139761090996096 model_lib_v2.py:652] Step 230400 per-step time 0.236s loss=0.815
    INFO:tensorflow:Step 230500 per-step time 0.258s loss=0.879
    I1011 03:03:07.963582 139761090996096 model_lib_v2.py:652] Step 230500 per-step time 0.258s loss=0.879
    INFO:tensorflow:Step 230600 per-step time 0.243s loss=0.801
    I1011 03:03:33.133035 139761090996096 model_lib_v2.py:652] Step 230600 per-step time 0.243s loss=0.801
    INFO:tensorflow:Step 230700 per-step time 0.252s loss=0.482
    I1011 03:03:58.046793 139761090996096 model_lib_v2.py:652] Step 230700 per-step time 0.252s loss=0.482
    INFO:tensorflow:Step 230800 per-step time 0.256s loss=0.517
    I1011 03:04:22.917856 139761090996096 model_lib_v2.py:652] Step 230800 per-step time 0.256s loss=0.517
    INFO:tensorflow:Step 230900 per-step time 0.244s loss=0.514
    I1011 03:04:47.824757 139761090996096 model_lib_v2.py:652] Step 230900 per-step time 0.244s loss=0.514
    INFO:tensorflow:Step 231000 per-step time 0.251s loss=0.623
    I1011 03:05:12.603476 139761090996096 model_lib_v2.py:652] Step 231000 per-step time 0.251s loss=0.623
    INFO:tensorflow:Step 231100 per-step time 0.242s loss=0.608
    I1011 03:05:38.288311 139761090996096 model_lib_v2.py:652] Step 231100 per-step time 0.242s loss=0.608
    INFO:tensorflow:Step 231200 per-step time 0.267s loss=0.840
    I1011 03:06:03.276473 139761090996096 model_lib_v2.py:652] Step 231200 per-step time 0.267s loss=0.840
    INFO:tensorflow:Step 231300 per-step time 0.232s loss=1.060
    I1011 03:06:28.156030 139761090996096 model_lib_v2.py:652] Step 231300 per-step time 0.232s loss=1.060
    INFO:tensorflow:Step 231400 per-step time 0.238s loss=0.856
    I1011 03:06:53.027303 139761090996096 model_lib_v2.py:652] Step 231400 per-step time 0.238s loss=0.856
    INFO:tensorflow:Step 231500 per-step time 0.250s loss=0.672
    I1011 03:07:17.847660 139761090996096 model_lib_v2.py:652] Step 231500 per-step time 0.250s loss=0.672
    INFO:tensorflow:Step 231600 per-step time 0.260s loss=0.660
    I1011 03:07:42.607794 139761090996096 model_lib_v2.py:652] Step 231600 per-step time 0.260s loss=0.660
    INFO:tensorflow:Step 231700 per-step time 0.261s loss=0.661
    I1011 03:08:07.475841 139761090996096 model_lib_v2.py:652] Step 231700 per-step time 0.261s loss=0.661
    INFO:tensorflow:Step 231800 per-step time 0.249s loss=0.720
    I1011 03:08:32.490940 139761090996096 model_lib_v2.py:652] Step 231800 per-step time 0.249s loss=0.720
    INFO:tensorflow:Step 231900 per-step time 0.246s loss=0.734
    I1011 03:08:57.341694 139761090996096 model_lib_v2.py:652] Step 231900 per-step time 0.246s loss=0.734
    INFO:tensorflow:Step 232000 per-step time 0.245s loss=0.506
    I1011 03:09:22.196784 139761090996096 model_lib_v2.py:652] Step 232000 per-step time 0.245s loss=0.506
    INFO:tensorflow:Step 232100 per-step time 0.239s loss=0.454
    I1011 03:09:47.776881 139761090996096 model_lib_v2.py:652] Step 232100 per-step time 0.239s loss=0.454
    INFO:tensorflow:Step 232200 per-step time 0.243s loss=0.474
    I1011 03:10:12.680451 139761090996096 model_lib_v2.py:652] Step 232200 per-step time 0.243s loss=0.474
    INFO:tensorflow:Step 232300 per-step time 0.243s loss=0.632
    I1011 03:10:37.565008 139761090996096 model_lib_v2.py:652] Step 232300 per-step time 0.243s loss=0.632
    INFO:tensorflow:Step 232400 per-step time 0.240s loss=0.749
    I1011 03:11:02.460431 139761090996096 model_lib_v2.py:652] Step 232400 per-step time 0.240s loss=0.749
    INFO:tensorflow:Step 232500 per-step time 0.237s loss=0.805
    I1011 03:11:27.240862 139761090996096 model_lib_v2.py:652] Step 232500 per-step time 0.237s loss=0.805
    INFO:tensorflow:Step 232600 per-step time 0.238s loss=0.450
    I1011 03:11:52.068912 139761090996096 model_lib_v2.py:652] Step 232600 per-step time 0.238s loss=0.450
    INFO:tensorflow:Step 232700 per-step time 0.247s loss=0.485
    I1011 03:12:16.750580 139761090996096 model_lib_v2.py:652] Step 232700 per-step time 0.247s loss=0.485
    INFO:tensorflow:Step 232800 per-step time 0.253s loss=0.645
    I1011 03:12:41.508052 139761090996096 model_lib_v2.py:652] Step 232800 per-step time 0.253s loss=0.645
    INFO:tensorflow:Step 232900 per-step time 0.248s loss=0.960
    I1011 03:13:06.415328 139761090996096 model_lib_v2.py:652] Step 232900 per-step time 0.248s loss=0.960
    INFO:tensorflow:Step 233000 per-step time 0.254s loss=0.583
    I1011 03:13:31.276140 139761090996096 model_lib_v2.py:652] Step 233000 per-step time 0.254s loss=0.583
    INFO:tensorflow:Step 233100 per-step time 0.249s loss=0.643
    I1011 03:13:57.156643 139761090996096 model_lib_v2.py:652] Step 233100 per-step time 0.249s loss=0.643
    INFO:tensorflow:Step 233200 per-step time 0.245s loss=0.791
    I1011 03:14:21.950176 139761090996096 model_lib_v2.py:652] Step 233200 per-step time 0.245s loss=0.791
    INFO:tensorflow:Step 233300 per-step time 0.248s loss=0.774
    I1011 03:14:46.846289 139761090996096 model_lib_v2.py:652] Step 233300 per-step time 0.248s loss=0.774
    INFO:tensorflow:Step 233400 per-step time 0.243s loss=0.941
    I1011 03:15:11.745937 139761090996096 model_lib_v2.py:652] Step 233400 per-step time 0.243s loss=0.941
    INFO:tensorflow:Step 233500 per-step time 0.271s loss=0.576
    I1011 03:15:36.585045 139761090996096 model_lib_v2.py:652] Step 233500 per-step time 0.271s loss=0.576
    INFO:tensorflow:Step 233600 per-step time 0.256s loss=0.700
    I1011 03:16:01.370745 139761090996096 model_lib_v2.py:652] Step 233600 per-step time 0.256s loss=0.700
    INFO:tensorflow:Step 233700 per-step time 0.242s loss=0.612
    I1011 03:16:26.257895 139761090996096 model_lib_v2.py:652] Step 233700 per-step time 0.242s loss=0.612
    INFO:tensorflow:Step 233800 per-step time 0.245s loss=0.734
    I1011 03:16:51.178529 139761090996096 model_lib_v2.py:652] Step 233800 per-step time 0.245s loss=0.734
    INFO:tensorflow:Step 233900 per-step time 0.248s loss=0.635
    I1011 03:17:16.157902 139761090996096 model_lib_v2.py:652] Step 233900 per-step time 0.248s loss=0.635
    INFO:tensorflow:Step 234000 per-step time 0.243s loss=0.607
    I1011 03:17:41.099239 139761090996096 model_lib_v2.py:652] Step 234000 per-step time 0.243s loss=0.607
    INFO:tensorflow:Step 234100 per-step time 0.247s loss=0.583
    I1011 03:18:06.808758 139761090996096 model_lib_v2.py:652] Step 234100 per-step time 0.247s loss=0.583
    INFO:tensorflow:Step 234200 per-step time 0.235s loss=0.721
    I1011 03:18:31.776912 139761090996096 model_lib_v2.py:652] Step 234200 per-step time 0.235s loss=0.721
    INFO:tensorflow:Step 234300 per-step time 0.236s loss=0.554
    I1011 03:18:56.801238 139761090996096 model_lib_v2.py:652] Step 234300 per-step time 0.236s loss=0.554
    INFO:tensorflow:Step 234400 per-step time 0.251s loss=0.485
    I1011 03:19:21.833346 139761090996096 model_lib_v2.py:652] Step 234400 per-step time 0.251s loss=0.485
    INFO:tensorflow:Step 234500 per-step time 0.253s loss=0.651
    I1011 03:19:46.702029 139761090996096 model_lib_v2.py:652] Step 234500 per-step time 0.253s loss=0.651
    INFO:tensorflow:Step 234600 per-step time 0.241s loss=0.449
    I1011 03:20:11.573779 139761090996096 model_lib_v2.py:652] Step 234600 per-step time 0.241s loss=0.449
    INFO:tensorflow:Step 234700 per-step time 0.258s loss=0.614
    I1011 03:20:36.472444 139761090996096 model_lib_v2.py:652] Step 234700 per-step time 0.258s loss=0.614
    INFO:tensorflow:Step 234800 per-step time 0.243s loss=0.593
    I1011 03:21:01.371443 139761090996096 model_lib_v2.py:652] Step 234800 per-step time 0.243s loss=0.593
    INFO:tensorflow:Step 234900 per-step time 0.243s loss=1.026
    I1011 03:21:26.269698 139761090996096 model_lib_v2.py:652] Step 234900 per-step time 0.243s loss=1.026
    INFO:tensorflow:Step 235000 per-step time 0.247s loss=0.393
    I1011 03:21:51.266696 139761090996096 model_lib_v2.py:652] Step 235000 per-step time 0.247s loss=0.393
    INFO:tensorflow:Step 235100 per-step time 0.253s loss=0.761
    I1011 03:22:17.196080 139761090996096 model_lib_v2.py:652] Step 235100 per-step time 0.253s loss=0.761
    INFO:tensorflow:Step 235200 per-step time 0.246s loss=0.866
    I1011 03:22:42.218789 139761090996096 model_lib_v2.py:652] Step 235200 per-step time 0.246s loss=0.866
    INFO:tensorflow:Step 235300 per-step time 0.245s loss=0.750
    I1011 03:23:07.194389 139761090996096 model_lib_v2.py:652] Step 235300 per-step time 0.245s loss=0.750
    INFO:tensorflow:Step 235400 per-step time 0.240s loss=0.543
    I1011 03:23:32.066510 139761090996096 model_lib_v2.py:652] Step 235400 per-step time 0.240s loss=0.543
    INFO:tensorflow:Step 235500 per-step time 0.247s loss=0.573
    I1011 03:23:57.004146 139761090996096 model_lib_v2.py:652] Step 235500 per-step time 0.247s loss=0.573
    INFO:tensorflow:Step 235600 per-step time 0.247s loss=0.629
    I1011 03:24:21.985414 139761090996096 model_lib_v2.py:652] Step 235600 per-step time 0.247s loss=0.629
    INFO:tensorflow:Step 235700 per-step time 0.247s loss=0.535
    I1011 03:24:47.037872 139761090996096 model_lib_v2.py:652] Step 235700 per-step time 0.247s loss=0.535
    INFO:tensorflow:Step 235800 per-step time 0.280s loss=1.035
    I1011 03:25:12.165085 139761090996096 model_lib_v2.py:652] Step 235800 per-step time 0.280s loss=1.035
    INFO:tensorflow:Step 235900 per-step time 0.238s loss=0.561
    I1011 03:25:37.110783 139761090996096 model_lib_v2.py:652] Step 235900 per-step time 0.238s loss=0.561
    INFO:tensorflow:Step 236000 per-step time 0.256s loss=0.710
    I1011 03:26:02.079370 139761090996096 model_lib_v2.py:652] Step 236000 per-step time 0.256s loss=0.710
    INFO:tensorflow:Step 236100 per-step time 0.245s loss=0.747
    I1011 03:26:27.965075 139761090996096 model_lib_v2.py:652] Step 236100 per-step time 0.245s loss=0.747
    INFO:tensorflow:Step 236200 per-step time 0.249s loss=0.823
    I1011 03:26:52.960054 139761090996096 model_lib_v2.py:652] Step 236200 per-step time 0.249s loss=0.823
    INFO:tensorflow:Step 236300 per-step time 0.249s loss=0.659
    I1011 03:27:17.958003 139761090996096 model_lib_v2.py:652] Step 236300 per-step time 0.249s loss=0.659
    INFO:tensorflow:Step 236400 per-step time 0.257s loss=0.806
    I1011 03:27:42.877974 139761090996096 model_lib_v2.py:652] Step 236400 per-step time 0.257s loss=0.806
    INFO:tensorflow:Step 236500 per-step time 0.255s loss=0.651
    I1011 03:28:07.836258 139761090996096 model_lib_v2.py:652] Step 236500 per-step time 0.255s loss=0.651
    INFO:tensorflow:Step 236600 per-step time 0.263s loss=0.930
    I1011 03:28:32.899905 139761090996096 model_lib_v2.py:652] Step 236600 per-step time 0.263s loss=0.930
    INFO:tensorflow:Step 236700 per-step time 0.241s loss=0.611
    I1011 03:28:57.782701 139761090996096 model_lib_v2.py:652] Step 236700 per-step time 0.241s loss=0.611
    INFO:tensorflow:Step 236800 per-step time 0.253s loss=0.681
    I1011 03:29:22.883225 139761090996096 model_lib_v2.py:652] Step 236800 per-step time 0.253s loss=0.681
    INFO:tensorflow:Step 236900 per-step time 0.253s loss=1.046
    I1011 03:29:47.912795 139761090996096 model_lib_v2.py:652] Step 236900 per-step time 0.253s loss=1.046
    INFO:tensorflow:Step 237000 per-step time 0.243s loss=0.427
    I1011 03:30:12.844481 139761090996096 model_lib_v2.py:652] Step 237000 per-step time 0.243s loss=0.427
    INFO:tensorflow:Step 237100 per-step time 0.244s loss=0.681
    I1011 03:30:39.042939 139761090996096 model_lib_v2.py:652] Step 237100 per-step time 0.244s loss=0.681
    INFO:tensorflow:Step 237200 per-step time 0.263s loss=0.668
    I1011 03:31:03.960469 139761090996096 model_lib_v2.py:652] Step 237200 per-step time 0.263s loss=0.668
    INFO:tensorflow:Step 237300 per-step time 0.240s loss=0.901
    I1011 03:31:29.138647 139761090996096 model_lib_v2.py:652] Step 237300 per-step time 0.240s loss=0.901
    INFO:tensorflow:Step 237400 per-step time 0.238s loss=0.885
    I1011 03:31:54.152407 139761090996096 model_lib_v2.py:652] Step 237400 per-step time 0.238s loss=0.885
    INFO:tensorflow:Step 237500 per-step time 0.244s loss=0.479
    I1011 03:32:19.347834 139761090996096 model_lib_v2.py:652] Step 237500 per-step time 0.244s loss=0.479
    INFO:tensorflow:Step 237600 per-step time 0.245s loss=0.647
    I1011 03:32:44.344326 139761090996096 model_lib_v2.py:652] Step 237600 per-step time 0.245s loss=0.647
    INFO:tensorflow:Step 237700 per-step time 0.240s loss=0.743
    I1011 03:33:09.382091 139761090996096 model_lib_v2.py:652] Step 237700 per-step time 0.240s loss=0.743
    INFO:tensorflow:Step 237800 per-step time 0.265s loss=0.762
    I1011 03:33:34.324860 139761090996096 model_lib_v2.py:652] Step 237800 per-step time 0.265s loss=0.762
    INFO:tensorflow:Step 237900 per-step time 0.235s loss=0.682
    I1011 03:33:59.148835 139761090996096 model_lib_v2.py:652] Step 237900 per-step time 0.235s loss=0.682
    INFO:tensorflow:Step 238000 per-step time 0.243s loss=0.405
    I1011 03:34:24.204610 139761090996096 model_lib_v2.py:652] Step 238000 per-step time 0.243s loss=0.405
    INFO:tensorflow:Step 238100 per-step time 0.248s loss=0.950
    I1011 03:34:50.178780 139761090996096 model_lib_v2.py:652] Step 238100 per-step time 0.248s loss=0.950
    INFO:tensorflow:Step 238200 per-step time 0.237s loss=0.810
    I1011 03:35:15.142280 139761090996096 model_lib_v2.py:652] Step 238200 per-step time 0.237s loss=0.810
    INFO:tensorflow:Step 238300 per-step time 0.238s loss=0.482
    I1011 03:35:40.103083 139761090996096 model_lib_v2.py:652] Step 238300 per-step time 0.238s loss=0.482
    INFO:tensorflow:Step 238400 per-step time 0.260s loss=0.514
    I1011 03:36:04.990795 139761090996096 model_lib_v2.py:652] Step 238400 per-step time 0.260s loss=0.514
    INFO:tensorflow:Step 238500 per-step time 0.246s loss=0.731
    I1011 03:36:30.066960 139761090996096 model_lib_v2.py:652] Step 238500 per-step time 0.246s loss=0.731
    INFO:tensorflow:Step 238600 per-step time 0.250s loss=0.557
    I1011 03:36:55.110245 139761090996096 model_lib_v2.py:652] Step 238600 per-step time 0.250s loss=0.557
    INFO:tensorflow:Step 238700 per-step time 0.250s loss=0.741
    I1011 03:37:20.169090 139761090996096 model_lib_v2.py:652] Step 238700 per-step time 0.250s loss=0.741
    INFO:tensorflow:Step 238800 per-step time 0.256s loss=0.598
    I1011 03:37:45.242340 139761090996096 model_lib_v2.py:652] Step 238800 per-step time 0.256s loss=0.598
    INFO:tensorflow:Step 238900 per-step time 0.255s loss=0.853
    I1011 03:38:10.194183 139761090996096 model_lib_v2.py:652] Step 238900 per-step time 0.255s loss=0.853
    INFO:tensorflow:Step 239000 per-step time 0.248s loss=0.558
    I1011 03:38:35.134134 139761090996096 model_lib_v2.py:652] Step 239000 per-step time 0.248s loss=0.558
    INFO:tensorflow:Step 239100 per-step time 0.250s loss=0.920
    I1011 03:39:00.932140 139761090996096 model_lib_v2.py:652] Step 239100 per-step time 0.250s loss=0.920
    INFO:tensorflow:Step 239200 per-step time 0.269s loss=0.715
    I1011 03:39:26.057298 139761090996096 model_lib_v2.py:652] Step 239200 per-step time 0.269s loss=0.715
    INFO:tensorflow:Step 239300 per-step time 0.251s loss=0.483
    I1011 03:39:51.329986 139761090996096 model_lib_v2.py:652] Step 239300 per-step time 0.251s loss=0.483
    INFO:tensorflow:Step 239400 per-step time 0.240s loss=0.542
    I1011 03:40:16.389589 139761090996096 model_lib_v2.py:652] Step 239400 per-step time 0.240s loss=0.542
    INFO:tensorflow:Step 239500 per-step time 0.248s loss=0.518
    I1011 03:40:41.464074 139761090996096 model_lib_v2.py:652] Step 239500 per-step time 0.248s loss=0.518
    INFO:tensorflow:Step 239600 per-step time 0.244s loss=0.939
    I1011 03:41:06.666309 139761090996096 model_lib_v2.py:652] Step 239600 per-step time 0.244s loss=0.939
    INFO:tensorflow:Step 239700 per-step time 0.266s loss=0.570
    I1011 03:41:31.702773 139761090996096 model_lib_v2.py:652] Step 239700 per-step time 0.266s loss=0.570
    INFO:tensorflow:Step 239800 per-step time 0.253s loss=0.460
    I1011 03:41:56.699468 139761090996096 model_lib_v2.py:652] Step 239800 per-step time 0.253s loss=0.460
    INFO:tensorflow:Step 239900 per-step time 0.254s loss=0.762
    I1011 03:42:21.850215 139761090996096 model_lib_v2.py:652] Step 239900 per-step time 0.254s loss=0.762
    INFO:tensorflow:Step 240000 per-step time 0.262s loss=0.446
    I1011 03:42:46.829993 139761090996096 model_lib_v2.py:652] Step 240000 per-step time 0.262s loss=0.446
    INFO:tensorflow:Step 240100 per-step time 0.258s loss=0.657
    I1011 03:43:12.654613 139761090996096 model_lib_v2.py:652] Step 240100 per-step time 0.258s loss=0.657
    INFO:tensorflow:Step 240200 per-step time 0.242s loss=0.663
    I1011 03:43:37.764045 139761090996096 model_lib_v2.py:652] Step 240200 per-step time 0.242s loss=0.663
    INFO:tensorflow:Step 240300 per-step time 0.248s loss=0.803
    I1011 03:44:02.830036 139761090996096 model_lib_v2.py:652] Step 240300 per-step time 0.248s loss=0.803
    INFO:tensorflow:Step 240400 per-step time 0.256s loss=0.467
    I1011 03:44:27.832111 139761090996096 model_lib_v2.py:652] Step 240400 per-step time 0.256s loss=0.467
    INFO:tensorflow:Step 240500 per-step time 0.237s loss=0.616
    I1011 03:44:52.890780 139761090996096 model_lib_v2.py:652] Step 240500 per-step time 0.237s loss=0.616
    INFO:tensorflow:Step 240600 per-step time 0.265s loss=0.487
    I1011 03:45:18.034033 139761090996096 model_lib_v2.py:652] Step 240600 per-step time 0.265s loss=0.487
    INFO:tensorflow:Step 240700 per-step time 0.251s loss=0.897
    I1011 03:45:43.052293 139761090996096 model_lib_v2.py:652] Step 240700 per-step time 0.251s loss=0.897
    INFO:tensorflow:Step 240800 per-step time 0.243s loss=0.753
    I1011 03:46:08.018157 139761090996096 model_lib_v2.py:652] Step 240800 per-step time 0.243s loss=0.753
    INFO:tensorflow:Step 240900 per-step time 0.246s loss=0.801
    I1011 03:46:33.056903 139761090996096 model_lib_v2.py:652] Step 240900 per-step time 0.246s loss=0.801
    INFO:tensorflow:Step 241000 per-step time 0.260s loss=0.537
    I1011 03:46:58.092156 139761090996096 model_lib_v2.py:652] Step 241000 per-step time 0.260s loss=0.537
    INFO:tensorflow:Step 241100 per-step time 0.239s loss=0.587
    I1011 03:47:24.040847 139761090996096 model_lib_v2.py:652] Step 241100 per-step time 0.239s loss=0.587
    INFO:tensorflow:Step 241200 per-step time 0.252s loss=0.502
    I1011 03:47:49.088674 139761090996096 model_lib_v2.py:652] Step 241200 per-step time 0.252s loss=0.502
    INFO:tensorflow:Step 241300 per-step time 0.259s loss=0.531
    I1011 03:48:14.133425 139761090996096 model_lib_v2.py:652] Step 241300 per-step time 0.259s loss=0.531
    INFO:tensorflow:Step 241400 per-step time 0.254s loss=0.502
    I1011 03:48:39.132857 139761090996096 model_lib_v2.py:652] Step 241400 per-step time 0.254s loss=0.502
    INFO:tensorflow:Step 241500 per-step time 0.250s loss=0.779
    I1011 03:49:04.255936 139761090996096 model_lib_v2.py:652] Step 241500 per-step time 0.250s loss=0.779
    INFO:tensorflow:Step 241600 per-step time 0.262s loss=0.667
    I1011 03:49:29.113574 139761090996096 model_lib_v2.py:652] Step 241600 per-step time 0.262s loss=0.667
    INFO:tensorflow:Step 241700 per-step time 0.246s loss=0.556
    I1011 03:49:54.239963 139761090996096 model_lib_v2.py:652] Step 241700 per-step time 0.246s loss=0.556
    INFO:tensorflow:Step 241800 per-step time 0.257s loss=0.414
    I1011 03:50:19.225754 139761090996096 model_lib_v2.py:652] Step 241800 per-step time 0.257s loss=0.414
    INFO:tensorflow:Step 241900 per-step time 0.238s loss=0.555
    I1011 03:50:44.249299 139761090996096 model_lib_v2.py:652] Step 241900 per-step time 0.238s loss=0.555
    INFO:tensorflow:Step 242000 per-step time 0.263s loss=0.804
    I1011 03:51:09.308684 139761090996096 model_lib_v2.py:652] Step 242000 per-step time 0.263s loss=0.804
    INFO:tensorflow:Step 242100 per-step time 0.247s loss=0.567
    I1011 03:51:35.309885 139761090996096 model_lib_v2.py:652] Step 242100 per-step time 0.247s loss=0.567
    INFO:tensorflow:Step 242200 per-step time 0.246s loss=0.610
    I1011 03:52:00.192744 139761090996096 model_lib_v2.py:652] Step 242200 per-step time 0.246s loss=0.610
    INFO:tensorflow:Step 242300 per-step time 0.242s loss=0.738
    I1011 03:52:25.056086 139761090996096 model_lib_v2.py:652] Step 242300 per-step time 0.242s loss=0.738
    INFO:tensorflow:Step 242400 per-step time 0.249s loss=0.727
    I1011 03:52:50.064795 139761090996096 model_lib_v2.py:652] Step 242400 per-step time 0.249s loss=0.727
    INFO:tensorflow:Step 242500 per-step time 0.239s loss=0.604
    I1011 03:53:14.930124 139761090996096 model_lib_v2.py:652] Step 242500 per-step time 0.239s loss=0.604
    INFO:tensorflow:Step 242600 per-step time 0.242s loss=0.671
    I1011 03:53:39.921222 139761090996096 model_lib_v2.py:652] Step 242600 per-step time 0.242s loss=0.671
    INFO:tensorflow:Step 242700 per-step time 0.248s loss=0.848
    I1011 03:54:04.815115 139761090996096 model_lib_v2.py:652] Step 242700 per-step time 0.248s loss=0.848
    INFO:tensorflow:Step 242800 per-step time 0.243s loss=0.926
    I1011 03:54:29.789359 139761090996096 model_lib_v2.py:652] Step 242800 per-step time 0.243s loss=0.926
    INFO:tensorflow:Step 242900 per-step time 0.242s loss=0.575
    I1011 03:54:54.659368 139761090996096 model_lib_v2.py:652] Step 242900 per-step time 0.242s loss=0.575
    INFO:tensorflow:Step 243000 per-step time 0.254s loss=0.609
    I1011 03:55:19.766131 139761090996096 model_lib_v2.py:652] Step 243000 per-step time 0.254s loss=0.609
    INFO:tensorflow:Step 243100 per-step time 0.251s loss=0.587
    I1011 03:55:45.480639 139761090996096 model_lib_v2.py:652] Step 243100 per-step time 0.251s loss=0.587
    INFO:tensorflow:Step 243200 per-step time 0.248s loss=0.612
    I1011 03:56:10.475081 139761090996096 model_lib_v2.py:652] Step 243200 per-step time 0.248s loss=0.612
    INFO:tensorflow:Step 243300 per-step time 0.253s loss=0.486
    I1011 03:56:35.454867 139761090996096 model_lib_v2.py:652] Step 243300 per-step time 0.253s loss=0.486
    INFO:tensorflow:Step 243400 per-step time 0.240s loss=0.741
    I1011 03:57:00.376143 139761090996096 model_lib_v2.py:652] Step 243400 per-step time 0.240s loss=0.741
    INFO:tensorflow:Step 243500 per-step time 0.247s loss=0.678
    I1011 03:57:25.397093 139761090996096 model_lib_v2.py:652] Step 243500 per-step time 0.247s loss=0.678
    INFO:tensorflow:Step 243600 per-step time 0.249s loss=0.492
    I1011 03:57:50.478385 139761090996096 model_lib_v2.py:652] Step 243600 per-step time 0.249s loss=0.492
    INFO:tensorflow:Step 243700 per-step time 0.243s loss=0.786
    I1011 03:58:15.460549 139761090996096 model_lib_v2.py:652] Step 243700 per-step time 0.243s loss=0.786
    INFO:tensorflow:Step 243800 per-step time 0.256s loss=0.620
    I1011 03:58:40.397969 139761090996096 model_lib_v2.py:652] Step 243800 per-step time 0.256s loss=0.620
    INFO:tensorflow:Step 243900 per-step time 0.247s loss=0.381
    I1011 03:59:05.311009 139761090996096 model_lib_v2.py:652] Step 243900 per-step time 0.247s loss=0.381
    INFO:tensorflow:Step 244000 per-step time 0.257s loss=0.491
    I1011 03:59:30.095111 139761090996096 model_lib_v2.py:652] Step 244000 per-step time 0.257s loss=0.491
    INFO:tensorflow:Step 244100 per-step time 0.252s loss=0.923
    I1011 03:59:55.991617 139761090996096 model_lib_v2.py:652] Step 244100 per-step time 0.252s loss=0.923
    INFO:tensorflow:Step 244200 per-step time 0.246s loss=0.642
    I1011 04:00:21.192516 139761090996096 model_lib_v2.py:652] Step 244200 per-step time 0.246s loss=0.642
    INFO:tensorflow:Step 244300 per-step time 0.257s loss=0.708
    I1011 04:00:46.109291 139761090996096 model_lib_v2.py:652] Step 244300 per-step time 0.257s loss=0.708
    INFO:tensorflow:Step 244400 per-step time 0.245s loss=0.903
    I1011 04:01:11.091464 139761090996096 model_lib_v2.py:652] Step 244400 per-step time 0.245s loss=0.903
    INFO:tensorflow:Step 244500 per-step time 0.247s loss=0.645
    I1011 04:01:36.152883 139761090996096 model_lib_v2.py:652] Step 244500 per-step time 0.247s loss=0.645
    INFO:tensorflow:Step 244600 per-step time 0.247s loss=0.642
    I1011 04:02:01.207612 139761090996096 model_lib_v2.py:652] Step 244600 per-step time 0.247s loss=0.642
    INFO:tensorflow:Step 244700 per-step time 0.246s loss=0.848
    I1011 04:02:26.090423 139761090996096 model_lib_v2.py:652] Step 244700 per-step time 0.246s loss=0.848
    INFO:tensorflow:Step 244800 per-step time 0.245s loss=0.548
    I1011 04:02:51.059214 139761090996096 model_lib_v2.py:652] Step 244800 per-step time 0.245s loss=0.548
    INFO:tensorflow:Step 244900 per-step time 0.252s loss=0.486
    I1011 04:03:15.926653 139761090996096 model_lib_v2.py:652] Step 244900 per-step time 0.252s loss=0.486
    INFO:tensorflow:Step 245000 per-step time 0.259s loss=0.595
    I1011 04:03:40.915790 139761090996096 model_lib_v2.py:652] Step 245000 per-step time 0.259s loss=0.595
    INFO:tensorflow:Step 245100 per-step time 0.251s loss=0.690
    I1011 04:04:06.678130 139761090996096 model_lib_v2.py:652] Step 245100 per-step time 0.251s loss=0.690
    INFO:tensorflow:Step 245200 per-step time 0.265s loss=0.555
    I1011 04:04:31.701255 139761090996096 model_lib_v2.py:652] Step 245200 per-step time 0.265s loss=0.555
    INFO:tensorflow:Step 245300 per-step time 0.253s loss=0.993
    I1011 04:04:56.721085 139761090996096 model_lib_v2.py:652] Step 245300 per-step time 0.253s loss=0.993
    INFO:tensorflow:Step 245400 per-step time 0.252s loss=0.953
    I1011 04:05:21.617224 139761090996096 model_lib_v2.py:652] Step 245400 per-step time 0.252s loss=0.953
    INFO:tensorflow:Step 245500 per-step time 0.249s loss=0.652
    I1011 04:05:46.808578 139761090996096 model_lib_v2.py:652] Step 245500 per-step time 0.249s loss=0.652
    INFO:tensorflow:Step 245600 per-step time 0.255s loss=0.887
    I1011 04:06:11.633915 139761090996096 model_lib_v2.py:652] Step 245600 per-step time 0.255s loss=0.887
    INFO:tensorflow:Step 245700 per-step time 0.257s loss=0.782
    I1011 04:06:36.603683 139761090996096 model_lib_v2.py:652] Step 245700 per-step time 0.257s loss=0.782
    INFO:tensorflow:Step 245800 per-step time 0.239s loss=0.661
    I1011 04:07:01.538398 139761090996096 model_lib_v2.py:652] Step 245800 per-step time 0.239s loss=0.661
    INFO:tensorflow:Step 245900 per-step time 0.251s loss=0.623
    I1011 04:07:26.378729 139761090996096 model_lib_v2.py:652] Step 245900 per-step time 0.251s loss=0.623
    INFO:tensorflow:Step 246000 per-step time 0.230s loss=0.483
    I1011 04:07:51.207556 139761090996096 model_lib_v2.py:652] Step 246000 per-step time 0.230s loss=0.483
    INFO:tensorflow:Step 246100 per-step time 0.252s loss=0.804
    I1011 04:08:17.061225 139761090996096 model_lib_v2.py:652] Step 246100 per-step time 0.252s loss=0.804
    INFO:tensorflow:Step 246200 per-step time 0.242s loss=0.699
    I1011 04:08:41.849338 139761090996096 model_lib_v2.py:652] Step 246200 per-step time 0.242s loss=0.699
    INFO:tensorflow:Step 246300 per-step time 0.246s loss=0.840
    I1011 04:09:06.693641 139761090996096 model_lib_v2.py:652] Step 246300 per-step time 0.246s loss=0.840
    INFO:tensorflow:Step 246400 per-step time 0.244s loss=0.652
    I1011 04:09:31.413076 139761090996096 model_lib_v2.py:652] Step 246400 per-step time 0.244s loss=0.652
    INFO:tensorflow:Step 246500 per-step time 0.244s loss=0.588
    I1011 04:09:56.352129 139761090996096 model_lib_v2.py:652] Step 246500 per-step time 0.244s loss=0.588
    INFO:tensorflow:Step 246600 per-step time 0.249s loss=0.520
    I1011 04:10:21.206505 139761090996096 model_lib_v2.py:652] Step 246600 per-step time 0.249s loss=0.520
    INFO:tensorflow:Step 246700 per-step time 0.253s loss=0.351
    I1011 04:10:46.208608 139761090996096 model_lib_v2.py:652] Step 246700 per-step time 0.253s loss=0.351
    INFO:tensorflow:Step 246800 per-step time 0.249s loss=0.520
    I1011 04:11:10.931970 139761090996096 model_lib_v2.py:652] Step 246800 per-step time 0.249s loss=0.520
    INFO:tensorflow:Step 246900 per-step time 0.256s loss=0.709
    I1011 04:11:35.757922 139761090996096 model_lib_v2.py:652] Step 246900 per-step time 0.256s loss=0.709
    INFO:tensorflow:Step 247000 per-step time 0.265s loss=0.691
    I1011 04:12:00.632194 139761090996096 model_lib_v2.py:652] Step 247000 per-step time 0.265s loss=0.691
    INFO:tensorflow:Step 247100 per-step time 0.257s loss=0.592
    I1011 04:12:26.319887 139761090996096 model_lib_v2.py:652] Step 247100 per-step time 0.257s loss=0.592
    INFO:tensorflow:Step 247200 per-step time 0.248s loss=0.896
    I1011 04:12:51.318118 139761090996096 model_lib_v2.py:652] Step 247200 per-step time 0.248s loss=0.896
    INFO:tensorflow:Step 247300 per-step time 0.244s loss=0.607
    I1011 04:13:16.267519 139761090996096 model_lib_v2.py:652] Step 247300 per-step time 0.244s loss=0.607
    INFO:tensorflow:Step 247400 per-step time 0.245s loss=0.378
    I1011 04:13:41.220549 139761090996096 model_lib_v2.py:652] Step 247400 per-step time 0.245s loss=0.378
    INFO:tensorflow:Step 247500 per-step time 0.235s loss=0.472
    I1011 04:14:06.206770 139761090996096 model_lib_v2.py:652] Step 247500 per-step time 0.235s loss=0.472
    INFO:tensorflow:Step 247600 per-step time 0.256s loss=0.606
    I1011 04:14:31.117633 139761090996096 model_lib_v2.py:652] Step 247600 per-step time 0.256s loss=0.606
    INFO:tensorflow:Step 247700 per-step time 0.261s loss=0.677
    I1011 04:14:55.949015 139761090996096 model_lib_v2.py:652] Step 247700 per-step time 0.261s loss=0.677
    INFO:tensorflow:Step 247800 per-step time 0.266s loss=0.716
    I1011 04:15:20.995676 139761090996096 model_lib_v2.py:652] Step 247800 per-step time 0.266s loss=0.716
    INFO:tensorflow:Step 247900 per-step time 0.262s loss=0.939
    I1011 04:15:45.908572 139761090996096 model_lib_v2.py:652] Step 247900 per-step time 0.262s loss=0.939
    INFO:tensorflow:Step 248000 per-step time 0.242s loss=0.655
    I1011 04:16:10.897796 139761090996096 model_lib_v2.py:652] Step 248000 per-step time 0.242s loss=0.655
    INFO:tensorflow:Step 248100 per-step time 0.246s loss=1.096
    I1011 04:16:36.554438 139761090996096 model_lib_v2.py:652] Step 248100 per-step time 0.246s loss=1.096
    INFO:tensorflow:Step 248200 per-step time 0.245s loss=0.513
    I1011 04:17:01.289751 139761090996096 model_lib_v2.py:652] Step 248200 per-step time 0.245s loss=0.513
    INFO:tensorflow:Step 248300 per-step time 0.251s loss=0.605
    I1011 04:17:26.257550 139761090996096 model_lib_v2.py:652] Step 248300 per-step time 0.251s loss=0.605
    INFO:tensorflow:Step 248400 per-step time 0.257s loss=0.722
    I1011 04:17:51.047286 139761090996096 model_lib_v2.py:652] Step 248400 per-step time 0.257s loss=0.722
    INFO:tensorflow:Step 248500 per-step time 0.240s loss=0.520
    I1011 04:18:15.845034 139761090996096 model_lib_v2.py:652] Step 248500 per-step time 0.240s loss=0.520
    INFO:tensorflow:Step 248600 per-step time 0.259s loss=0.894
    I1011 04:18:40.800220 139761090996096 model_lib_v2.py:652] Step 248600 per-step time 0.259s loss=0.894
    INFO:tensorflow:Step 248700 per-step time 0.240s loss=0.785
    I1011 04:19:05.740575 139761090996096 model_lib_v2.py:652] Step 248700 per-step time 0.240s loss=0.785
    INFO:tensorflow:Step 248800 per-step time 0.245s loss=0.748
    I1011 04:19:30.681418 139761090996096 model_lib_v2.py:652] Step 248800 per-step time 0.245s loss=0.748
    INFO:tensorflow:Step 248900 per-step time 0.247s loss=0.645
    I1011 04:19:55.559183 139761090996096 model_lib_v2.py:652] Step 248900 per-step time 0.247s loss=0.645
    INFO:tensorflow:Step 249000 per-step time 0.248s loss=0.463
    I1011 04:20:20.476361 139761090996096 model_lib_v2.py:652] Step 249000 per-step time 0.248s loss=0.463
    INFO:tensorflow:Step 249100 per-step time 0.256s loss=0.608
    I1011 04:20:46.312787 139761090996096 model_lib_v2.py:652] Step 249100 per-step time 0.256s loss=0.608
    INFO:tensorflow:Step 249200 per-step time 0.243s loss=0.729
    I1011 04:21:11.340941 139761090996096 model_lib_v2.py:652] Step 249200 per-step time 0.243s loss=0.729
    INFO:tensorflow:Step 249300 per-step time 0.244s loss=0.382
    I1011 04:21:36.064596 139761090996096 model_lib_v2.py:652] Step 249300 per-step time 0.244s loss=0.382
    INFO:tensorflow:Step 249400 per-step time 0.244s loss=0.531
    I1011 04:22:00.961120 139761090996096 model_lib_v2.py:652] Step 249400 per-step time 0.244s loss=0.531
    INFO:tensorflow:Step 249500 per-step time 0.251s loss=0.495
    I1011 04:22:25.736649 139761090996096 model_lib_v2.py:652] Step 249500 per-step time 0.251s loss=0.495
    INFO:tensorflow:Step 249600 per-step time 0.250s loss=0.892
    I1011 04:22:50.549240 139761090996096 model_lib_v2.py:652] Step 249600 per-step time 0.250s loss=0.892
    INFO:tensorflow:Step 249700 per-step time 0.256s loss=0.929
    I1011 04:23:15.386780 139761090996096 model_lib_v2.py:652] Step 249700 per-step time 0.256s loss=0.929
    INFO:tensorflow:Step 249800 per-step time 0.256s loss=0.565
    I1011 04:23:40.300064 139761090996096 model_lib_v2.py:652] Step 249800 per-step time 0.256s loss=0.565
    INFO:tensorflow:Step 249900 per-step time 0.253s loss=0.469
    I1011 04:24:05.171985 139761090996096 model_lib_v2.py:652] Step 249900 per-step time 0.253s loss=0.469
    INFO:tensorflow:Step 250000 per-step time 0.235s loss=0.577
    I1011 04:24:29.897685 139761090996096 model_lib_v2.py:652] Step 250000 per-step time 0.235s loss=0.577



```python
!zip myoutputmodel_250k.zip -r ./myoutputmodel
!cp myoutputmodel_250k.zip drive/My\ Drive/
```

      adding: myoutputmodel/ (stored 0%)
      adding: myoutputmodel/ckpt-246.index (deflated 81%)
      adding: myoutputmodel/ckpt-247.data-00000-of-00001 (deflated 11%)
      adding: myoutputmodel/ckpt-251.data-00000-of-00001 (deflated 11%)
      adding: myoutputmodel/ckpt-249.index (deflated 81%)
      adding: myoutputmodel/ckpt-249.data-00000-of-00001 (deflated 11%)
      adding: myoutputmodel/ckpt-248.index (deflated 81%)
      adding: myoutputmodel/checkpoint (deflated 75%)
      adding: myoutputmodel/ckpt-248.data-00000-of-00001 (deflated 11%)
      adding: myoutputmodel/ckpt-250.data-00000-of-00001 (deflated 11%)
      adding: myoutputmodel/train/ (stored 0%)
      adding: myoutputmodel/train/events.out.tfevents.1602222489.572f21843685.3333.1504.v2 (deflated 6%)
      adding: myoutputmodel/train/events.out.tfevents.1602365189.94b8267a15b6.765.1510.v2 (deflated 6%)
      adding: myoutputmodel/ckpt-251.index (deflated 81%)
      adding: myoutputmodel/ckpt-245.index (deflated 81%)
      adding: myoutputmodel/ckpt-245.data-00000-of-00001 (deflated 11%)
      adding: myoutputmodel/ckpt-246.data-00000-of-00001 (deflated 11%)
      adding: myoutputmodel/ckpt-250.index (deflated 81%)
      adding: myoutputmodel/ckpt-247.index (deflated 81%)


# More Evaluation at 250k steps


```python
!python models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={'./pipeline_file.config'} \
    --model_dir={'./myoutputmodel'} \
    --checkpoint_dir={'./myoutputmodel'} \
    --alsologtostderr
```

    2020-10-11 06:09:08.545238: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    WARNING:tensorflow:Forced number of epochs for all eval validations to be 1.
    W1011 06:09:14.740474 140459371468672 model_lib_v2.py:925] Forced number of epochs for all eval validations to be 1.
    INFO:tensorflow:Maybe overwriting sample_1_of_n_eval_examples: None
    I1011 06:09:14.740689 140459371468672 config_util.py:552] Maybe overwriting sample_1_of_n_eval_examples: None
    INFO:tensorflow:Maybe overwriting use_bfloat16: False
    I1011 06:09:14.740776 140459371468672 config_util.py:552] Maybe overwriting use_bfloat16: False
    INFO:tensorflow:Maybe overwriting eval_num_epochs: 1
    I1011 06:09:14.740841 140459371468672 config_util.py:552] Maybe overwriting eval_num_epochs: 1
    WARNING:tensorflow:Expected number of evaluation epochs is 1, but instead encountered `eval_on_train_input_config.num_epochs` = 0. Overwriting `num_epochs` to 1.
    W1011 06:09:14.740992 140459371468672 model_lib_v2.py:940] Expected number of evaluation epochs is 1, but instead encountered `eval_on_train_input_config.num_epochs` = 0. Overwriting `num_epochs` to 1.
    2020-10-11 06:09:14.799878: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
    2020-10-11 06:09:14.856735: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:09:14.857420: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
    coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.75GiB deviceMemoryBandwidth: 836.37GiB/s
    2020-10-11 06:09:14.857464: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-11 06:09:14.894664: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2020-10-11 06:09:15.004073: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
    2020-10-11 06:09:15.015938: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
    2020-10-11 06:09:15.068952: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
    2020-10-11 06:09:15.100422: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
    2020-10-11 06:09:15.224552: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    2020-10-11 06:09:15.224797: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:09:15.225532: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:09:15.226201: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
    2020-10-11 06:09:15.227204: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX512F
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2020-10-11 06:09:15.269975: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2000170000 Hz
    2020-10-11 06:09:15.270591: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x14d5640 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-10-11 06:09:15.270632: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-10-11 06:09:15.420475: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:09:15.421360: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x14d5800 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2020-10-11 06:09:15.421397: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
    2020-10-11 06:09:15.421825: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:09:15.422471: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
    coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.75GiB deviceMemoryBandwidth: 836.37GiB/s
    2020-10-11 06:09:15.422523: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-11 06:09:15.422592: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2020-10-11 06:09:15.422621: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
    2020-10-11 06:09:15.422652: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
    2020-10-11 06:09:15.422680: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
    2020-10-11 06:09:15.422722: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
    2020-10-11 06:09:15.422752: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    2020-10-11 06:09:15.422856: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:09:15.423523: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:09:15.424125: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
    2020-10-11 06:09:15.427634: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-11 06:09:19.281663: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-10-11 06:09:19.281738: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0 
    2020-10-11 06:09:19.281749: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N 
    2020-10-11 06:09:19.284746: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:09:19.285490: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:09:19.286100: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
    2020-10-11 06:09:19.286180: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14756 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0)
    INFO:tensorflow:Reading unweighted datasets: ['test.record']
    I1011 06:09:19.334519 140459371468672 dataset_builder.py:148] Reading unweighted datasets: ['test.record']
    INFO:tensorflow:Reading record datasets for input file: ['test.record']
    I1011 06:09:19.336271 140459371468672 dataset_builder.py:77] Reading record datasets for input file: ['test.record']
    INFO:tensorflow:Number of filenames to read: 1
    I1011 06:09:19.336463 140459371468672 dataset_builder.py:78] Number of filenames to read: 1
    WARNING:tensorflow:num_readers has been reduced to 1 to match input file shards.
    W1011 06:09:19.336532 140459371468672 dataset_builder.py:86] num_readers has been reduced to 1 to match input file shards.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:103: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.experimental_deterministic`.
    W1011 06:09:19.341277 140459371468672 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:103: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.experimental_deterministic`.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:222: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.map()
    W1011 06:09:19.422896 140459371468672 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:222: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.map()
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
    W1011 06:09:23.282591 140459371468672 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/inputs.py:262: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    W1011 06:09:24.737637 140459371468672 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/inputs.py:262: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    INFO:tensorflow:Waiting for new checkpoint at ./myoutputmodel
    I1011 06:09:27.474907 140459371468672 checkpoint_utils.py:125] Waiting for new checkpoint at ./myoutputmodel
    INFO:tensorflow:Found new checkpoint at ./myoutputmodel/ckpt-251
    I1011 06:09:27.487971 140459371468672 checkpoint_utils.py:134] Found new checkpoint at ./myoutputmodel/ckpt-251
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib_v2.py:702: set_learning_phase (from tensorflow.python.keras.backend) is deprecated and will be removed after 2020-10-11.
    Instructions for updating:
    Simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
    W1011 06:09:27.585626 140459371468672 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib_v2.py:702: set_learning_phase (from tensorflow.python.keras.backend) is deprecated and will be removed after 2020-10-11.
    Instructions for updating:
    Simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
    INFO:tensorflow:depth of additional conv before box predictor: 0
    I1011 06:09:35.455247 140459371468672 convolutional_keras_box_predictor.py:154] depth of additional conv before box predictor: 0
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/model_util.py:57: Tensor.experimental_ref (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use ref() instead.
    W1011 06:09:42.496448 140459371468672 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/model_util.py:57: Tensor.experimental_ref (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use ref() instead.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    W1011 06:09:49.093630 140459371468672 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/util/dispatch.py:201: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/eval_util.py:878: to_int64 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    W1011 06:09:55.184387 140459371468672 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/eval_util.py:878: to_int64 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    2020-10-11 06:10:02.073951: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2020-10-11 06:10:02.494760: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    INFO:tensorflow:Finished eval step 0
    I1011 06:10:05.099673 140459371468672 model_lib_v2.py:799] Finished eval step 0
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/visualization_utils.py:617: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    tf.py_func is deprecated in TF V2. Instead, there are two
        options available in V2.
        - tf.py_function takes a python function which manipulates tf eager
        tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
        an ndarray (just call tensor.numpy()) but having access to eager tensors
        means `tf.py_function`s can use accelerators such as GPUs as well as
        being differentiable using a gradient tape.
        - tf.numpy_function maintains the semantics of the deprecated tf.py_func
        (it is not differentiable, and manipulates numpy arrays). It drops the
        stateful argument making all functions stateful.
        
    W1011 06:10:05.264861 140459371468672 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/visualization_utils.py:617: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    tf.py_func is deprecated in TF V2. Instead, there are two
        options available in V2.
        - tf.py_function takes a python function which manipulates tf eager
        tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
        an ndarray (just call tensor.numpy()) but having access to eager tensors
        means `tf.py_function`s can use accelerators such as GPUs as well as
        being differentiable using a gradient tape.
        - tf.numpy_function maintains the semantics of the deprecated tf.py_func
        (it is not differentiable, and manipulates numpy arrays). It drops the
        stateful argument making all functions stateful.
        
    INFO:tensorflow:Finished eval step 100
    I1011 06:10:15.723350 140459371468672 model_lib_v2.py:799] Finished eval step 100
    INFO:tensorflow:Finished eval step 200
    I1011 06:10:22.808529 140459371468672 model_lib_v2.py:799] Finished eval step 200
    INFO:tensorflow:Finished eval step 300
    I1011 06:10:29.632488 140459371468672 model_lib_v2.py:799] Finished eval step 300
    INFO:tensorflow:Finished eval step 400
    I1011 06:10:36.651607 140459371468672 model_lib_v2.py:799] Finished eval step 400
    INFO:tensorflow:Finished eval step 500
    I1011 06:10:43.340469 140459371468672 model_lib_v2.py:799] Finished eval step 500
    INFO:tensorflow:Finished eval step 600
    I1011 06:10:50.330959 140459371468672 model_lib_v2.py:799] Finished eval step 600
    INFO:tensorflow:Finished eval step 700
    I1011 06:10:57.089128 140459371468672 model_lib_v2.py:799] Finished eval step 700
    INFO:tensorflow:Finished eval step 800
    I1011 06:11:03.818796 140459371468672 model_lib_v2.py:799] Finished eval step 800
    INFO:tensorflow:Finished eval step 900
    I1011 06:11:10.767721 140459371468672 model_lib_v2.py:799] Finished eval step 900
    INFO:tensorflow:Finished eval step 1000
    I1011 06:11:17.538134 140459371468672 model_lib_v2.py:799] Finished eval step 1000
    INFO:tensorflow:Finished eval step 1100
    I1011 06:11:24.218263 140459371468672 model_lib_v2.py:799] Finished eval step 1100
    INFO:tensorflow:Finished eval step 1200
    I1011 06:11:31.175992 140459371468672 model_lib_v2.py:799] Finished eval step 1200
    INFO:tensorflow:Finished eval step 1300
    I1011 06:11:38.367316 140459371468672 model_lib_v2.py:799] Finished eval step 1300
    INFO:tensorflow:Finished eval step 1400
    I1011 06:11:45.149812 140459371468672 model_lib_v2.py:799] Finished eval step 1400
    INFO:tensorflow:Finished eval step 1500
    I1011 06:11:52.032993 140459371468672 model_lib_v2.py:799] Finished eval step 1500
    INFO:tensorflow:Finished eval step 1600
    I1011 06:11:58.903228 140459371468672 model_lib_v2.py:799] Finished eval step 1600
    INFO:tensorflow:Finished eval step 1700
    I1011 06:12:05.748094 140459371468672 model_lib_v2.py:799] Finished eval step 1700
    INFO:tensorflow:Finished eval step 1800
    I1011 06:12:12.738460 140459371468672 model_lib_v2.py:799] Finished eval step 1800
    INFO:tensorflow:Finished eval step 1900
    I1011 06:12:19.473883 140459371468672 model_lib_v2.py:799] Finished eval step 1900
    INFO:tensorflow:Finished eval step 2000
    I1011 06:12:26.222066 140459371468672 model_lib_v2.py:799] Finished eval step 2000
    INFO:tensorflow:Finished eval step 2100
    I1011 06:12:33.028734 140459371468672 model_lib_v2.py:799] Finished eval step 2100
    INFO:tensorflow:Finished eval step 2200
    I1011 06:12:39.785356 140459371468672 model_lib_v2.py:799] Finished eval step 2200
    INFO:tensorflow:Finished eval step 2300
    I1011 06:12:46.506597 140459371468672 model_lib_v2.py:799] Finished eval step 2300
    INFO:tensorflow:Finished eval step 2400
    I1011 06:12:53.608750 140459371468672 model_lib_v2.py:799] Finished eval step 2400
    INFO:tensorflow:Finished eval step 2500
    I1011 06:13:00.466792 140459371468672 model_lib_v2.py:799] Finished eval step 2500
    INFO:tensorflow:Finished eval step 2600
    I1011 06:13:07.380937 140459371468672 model_lib_v2.py:799] Finished eval step 2600
    INFO:tensorflow:Finished eval step 2700
    I1011 06:13:14.239526 140459371468672 model_lib_v2.py:799] Finished eval step 2700
    INFO:tensorflow:Finished eval step 2800
    I1011 06:13:21.029787 140459371468672 model_lib_v2.py:799] Finished eval step 2800
    INFO:tensorflow:Finished eval step 2900
    I1011 06:13:27.866300 140459371468672 model_lib_v2.py:799] Finished eval step 2900
    INFO:tensorflow:Finished eval step 3000
    I1011 06:13:34.624782 140459371468672 model_lib_v2.py:799] Finished eval step 3000
    INFO:tensorflow:Finished eval step 3100
    I1011 06:13:41.268512 140459371468672 model_lib_v2.py:799] Finished eval step 3100
    INFO:tensorflow:Finished eval step 3200
    I1011 06:13:48.441150 140459371468672 model_lib_v2.py:799] Finished eval step 3200
    INFO:tensorflow:Finished eval step 3300
    I1011 06:13:55.141921 140459371468672 model_lib_v2.py:799] Finished eval step 3300
    INFO:tensorflow:Finished eval step 3400
    I1011 06:14:01.879700 140459371468672 model_lib_v2.py:799] Finished eval step 3400
    INFO:tensorflow:Finished eval step 3500
    I1011 06:14:08.565376 140459371468672 model_lib_v2.py:799] Finished eval step 3500
    INFO:tensorflow:Finished eval step 3600
    I1011 06:14:15.264399 140459371468672 model_lib_v2.py:799] Finished eval step 3600
    INFO:tensorflow:Finished eval step 3700
    I1011 06:14:21.861166 140459371468672 model_lib_v2.py:799] Finished eval step 3700
    INFO:tensorflow:Finished eval step 3800
    I1011 06:14:28.703341 140459371468672 model_lib_v2.py:799] Finished eval step 3800
    INFO:tensorflow:Finished eval step 3900
    I1011 06:14:35.518958 140459371468672 model_lib_v2.py:799] Finished eval step 3900
    INFO:tensorflow:Finished eval step 4000
    I1011 06:14:42.176605 140459371468672 model_lib_v2.py:799] Finished eval step 4000
    INFO:tensorflow:Finished eval step 4100
    I1011 06:14:49.524775 140459371468672 model_lib_v2.py:799] Finished eval step 4100
    INFO:tensorflow:Finished eval step 4200
    I1011 06:14:56.477955 140459371468672 model_lib_v2.py:799] Finished eval step 4200
    INFO:tensorflow:Finished eval step 4300
    I1011 06:15:03.863498 140459371468672 model_lib_v2.py:799] Finished eval step 4300
    INFO:tensorflow:Finished eval step 4400
    I1011 06:15:10.703170 140459371468672 model_lib_v2.py:799] Finished eval step 4400
    INFO:tensorflow:Finished eval step 4500
    I1011 06:15:17.379359 140459371468672 model_lib_v2.py:799] Finished eval step 4500
    INFO:tensorflow:Finished eval step 4600
    I1011 06:15:24.002530 140459371468672 model_lib_v2.py:799] Finished eval step 4600
    INFO:tensorflow:Finished eval step 4700
    I1011 06:15:30.618565 140459371468672 model_lib_v2.py:799] Finished eval step 4700
    INFO:tensorflow:Finished eval step 4800
    I1011 06:15:37.342270 140459371468672 model_lib_v2.py:799] Finished eval step 4800
    INFO:tensorflow:Finished eval step 4900
    I1011 06:15:43.961135 140459371468672 model_lib_v2.py:799] Finished eval step 4900
    INFO:tensorflow:Finished eval step 5000
    I1011 06:15:50.472046 140459371468672 model_lib_v2.py:799] Finished eval step 5000
    INFO:tensorflow:Finished eval step 5100
    I1011 06:15:57.142266 140459371468672 model_lib_v2.py:799] Finished eval step 5100
    INFO:tensorflow:Finished eval step 5200
    I1011 06:16:03.912544 140459371468672 model_lib_v2.py:799] Finished eval step 5200
    INFO:tensorflow:Finished eval step 5300
    I1011 06:16:11.406775 140459371468672 model_lib_v2.py:799] Finished eval step 5300
    INFO:tensorflow:Finished eval step 5400
    I1011 06:16:18.177784 140459371468672 model_lib_v2.py:799] Finished eval step 5400
    INFO:tensorflow:Finished eval step 5500
    I1011 06:16:24.971248 140459371468672 model_lib_v2.py:799] Finished eval step 5500
    INFO:tensorflow:Finished eval step 5600
    I1011 06:16:31.776851 140459371468672 model_lib_v2.py:799] Finished eval step 5600
    INFO:tensorflow:Finished eval step 5700
    I1011 06:16:38.563473 140459371468672 model_lib_v2.py:799] Finished eval step 5700
    INFO:tensorflow:Finished eval step 5800
    I1011 06:16:45.222795 140459371468672 model_lib_v2.py:799] Finished eval step 5800
    INFO:tensorflow:Finished eval step 5900
    I1011 06:16:51.940167 140459371468672 model_lib_v2.py:799] Finished eval step 5900
    INFO:tensorflow:Finished eval step 6000
    I1011 06:16:58.695565 140459371468672 model_lib_v2.py:799] Finished eval step 6000
    INFO:tensorflow:Finished eval step 6100
    I1011 06:17:05.372386 140459371468672 model_lib_v2.py:799] Finished eval step 6100
    INFO:tensorflow:Finished eval step 6200
    I1011 06:17:12.051581 140459371468672 model_lib_v2.py:799] Finished eval step 6200
    INFO:tensorflow:Finished eval step 6300
    I1011 06:17:18.766977 140459371468672 model_lib_v2.py:799] Finished eval step 6300
    INFO:tensorflow:Finished eval step 6400
    I1011 06:17:25.382531 140459371468672 model_lib_v2.py:799] Finished eval step 6400
    INFO:tensorflow:Finished eval step 6500
    I1011 06:17:32.261172 140459371468672 model_lib_v2.py:799] Finished eval step 6500
    INFO:tensorflow:Finished eval step 6600
    I1011 06:17:38.968465 140459371468672 model_lib_v2.py:799] Finished eval step 6600
    INFO:tensorflow:Finished eval step 6700
    I1011 06:17:45.717280 140459371468672 model_lib_v2.py:799] Finished eval step 6700
    INFO:tensorflow:Finished eval step 6800
    I1011 06:17:53.111688 140459371468672 model_lib_v2.py:799] Finished eval step 6800
    INFO:tensorflow:Finished eval step 6900
    I1011 06:17:59.845480 140459371468672 model_lib_v2.py:799] Finished eval step 6900
    INFO:tensorflow:Finished eval step 7000
    I1011 06:18:06.600927 140459371468672 model_lib_v2.py:799] Finished eval step 7000
    INFO:tensorflow:Finished eval step 7100
    I1011 06:18:13.316027 140459371468672 model_lib_v2.py:799] Finished eval step 7100
    INFO:tensorflow:Finished eval step 7200
    I1011 06:18:19.890566 140459371468672 model_lib_v2.py:799] Finished eval step 7200
    INFO:tensorflow:Finished eval step 7300
    I1011 06:18:26.590923 140459371468672 model_lib_v2.py:799] Finished eval step 7300
    INFO:tensorflow:Finished eval step 7400
    I1011 06:18:33.255172 140459371468672 model_lib_v2.py:799] Finished eval step 7400
    INFO:tensorflow:Finished eval step 7500
    I1011 06:18:39.865689 140459371468672 model_lib_v2.py:799] Finished eval step 7500
    INFO:tensorflow:Finished eval step 7600
    I1011 06:18:46.407981 140459371468672 model_lib_v2.py:799] Finished eval step 7600
    INFO:tensorflow:Finished eval step 7700
    I1011 06:18:52.881618 140459371468672 model_lib_v2.py:799] Finished eval step 7700
    INFO:tensorflow:Finished eval step 7800
    I1011 06:18:59.575151 140459371468672 model_lib_v2.py:799] Finished eval step 7800
    INFO:tensorflow:Finished eval step 7900
    I1011 06:19:06.341747 140459371468672 model_lib_v2.py:799] Finished eval step 7900
    INFO:tensorflow:Finished eval step 8000
    I1011 06:19:13.067039 140459371468672 model_lib_v2.py:799] Finished eval step 8000
    INFO:tensorflow:Finished eval step 8100
    I1011 06:19:19.818900 140459371468672 model_lib_v2.py:799] Finished eval step 8100
    INFO:tensorflow:Finished eval step 8200
    I1011 06:19:26.581202 140459371468672 model_lib_v2.py:799] Finished eval step 8200
    INFO:tensorflow:Finished eval step 8300
    I1011 06:19:33.286897 140459371468672 model_lib_v2.py:799] Finished eval step 8300
    INFO:tensorflow:Finished eval step 8400
    I1011 06:19:39.967193 140459371468672 model_lib_v2.py:799] Finished eval step 8400
    INFO:tensorflow:Finished eval step 8500
    I1011 06:19:46.554630 140459371468672 model_lib_v2.py:799] Finished eval step 8500
    INFO:tensorflow:Finished eval step 8600
    I1011 06:19:54.138476 140459371468672 model_lib_v2.py:799] Finished eval step 8600
    INFO:tensorflow:Finished eval step 8700
    I1011 06:20:00.747667 140459371468672 model_lib_v2.py:799] Finished eval step 8700
    INFO:tensorflow:Finished eval step 8800
    I1011 06:20:07.625803 140459371468672 model_lib_v2.py:799] Finished eval step 8800
    INFO:tensorflow:Finished eval step 8900
    I1011 06:20:14.781527 140459371468672 model_lib_v2.py:799] Finished eval step 8900
    INFO:tensorflow:Finished eval step 9000
    I1011 06:20:21.438864 140459371468672 model_lib_v2.py:799] Finished eval step 9000
    INFO:tensorflow:Finished eval step 9100
    I1011 06:20:28.148631 140459371468672 model_lib_v2.py:799] Finished eval step 9100
    INFO:tensorflow:Finished eval step 9200
    I1011 06:20:35.121044 140459371468672 model_lib_v2.py:799] Finished eval step 9200
    INFO:tensorflow:Finished eval step 9300
    I1011 06:20:41.911399 140459371468672 model_lib_v2.py:799] Finished eval step 9300
    INFO:tensorflow:Finished eval step 9400
    I1011 06:20:48.686941 140459371468672 model_lib_v2.py:799] Finished eval step 9400
    INFO:tensorflow:Finished eval step 9500
    I1011 06:20:55.499072 140459371468672 model_lib_v2.py:799] Finished eval step 9500
    INFO:tensorflow:Finished eval step 9600
    I1011 06:21:02.294083 140459371468672 model_lib_v2.py:799] Finished eval step 9600
    INFO:tensorflow:Finished eval step 9700
    I1011 06:21:09.183478 140459371468672 model_lib_v2.py:799] Finished eval step 9700
    INFO:tensorflow:Finished eval step 9800
    I1011 06:21:15.842357 140459371468672 model_lib_v2.py:799] Finished eval step 9800
    INFO:tensorflow:Finished eval step 9900
    I1011 06:21:22.509161 140459371468672 model_lib_v2.py:799] Finished eval step 9900
    INFO:tensorflow:Performing evaluation on 10000 images.
    I1011 06:21:28.982179 140459371468672 coco_evaluation.py:282] Performing evaluation on 10000 images.
    creating index...
    index created!
    INFO:tensorflow:Loading and preparing annotation results...
    I1011 06:21:29.122621 140459371468672 coco_tools.py:116] Loading and preparing annotation results...
    INFO:tensorflow:DONE (t=1.45s)
    I1011 06:21:30.577832 140459371468672 coco_tools.py:138] DONE (t=1.45s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=394.04s).
    Accumulating evaluation results...
    DONE (t=45.73s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.143
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.292
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.119
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.041
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.219
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.397
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.162
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.279
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.293
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.117
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.378
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.551
    INFO:tensorflow:Eval metrics at step 250000
    I1011 06:28:53.972208 140459371468672 model_lib_v2.py:853] Eval metrics at step 250000
    INFO:tensorflow:	+ DetectionBoxes_Precision/mAP: 0.142818
    I1011 06:28:53.982576 140459371468672 model_lib_v2.py:856] 	+ DetectionBoxes_Precision/mAP: 0.142818
    INFO:tensorflow:	+ DetectionBoxes_Precision/mAP@.50IOU: 0.292368
    I1011 06:28:53.984425 140459371468672 model_lib_v2.py:856] 	+ DetectionBoxes_Precision/mAP@.50IOU: 0.292368
    INFO:tensorflow:	+ DetectionBoxes_Precision/mAP@.75IOU: 0.119099
    I1011 06:28:53.985842 140459371468672 model_lib_v2.py:856] 	+ DetectionBoxes_Precision/mAP@.75IOU: 0.119099
    INFO:tensorflow:	+ DetectionBoxes_Precision/mAP (small): 0.040912
    I1011 06:28:53.987175 140459371468672 model_lib_v2.py:856] 	+ DetectionBoxes_Precision/mAP (small): 0.040912
    INFO:tensorflow:	+ DetectionBoxes_Precision/mAP (medium): 0.218648
    I1011 06:28:53.988506 140459371468672 model_lib_v2.py:856] 	+ DetectionBoxes_Precision/mAP (medium): 0.218648
    INFO:tensorflow:	+ DetectionBoxes_Precision/mAP (large): 0.397119
    I1011 06:28:53.989815 140459371468672 model_lib_v2.py:856] 	+ DetectionBoxes_Precision/mAP (large): 0.397119
    INFO:tensorflow:	+ DetectionBoxes_Recall/AR@1: 0.161660
    I1011 06:28:53.991139 140459371468672 model_lib_v2.py:856] 	+ DetectionBoxes_Recall/AR@1: 0.161660
    INFO:tensorflow:	+ DetectionBoxes_Recall/AR@10: 0.278956
    I1011 06:28:53.992369 140459371468672 model_lib_v2.py:856] 	+ DetectionBoxes_Recall/AR@10: 0.278956
    INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100: 0.293280
    I1011 06:28:53.993614 140459371468672 model_lib_v2.py:856] 	+ DetectionBoxes_Recall/AR@100: 0.293280
    INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100 (small): 0.116933
    I1011 06:28:53.994907 140459371468672 model_lib_v2.py:856] 	+ DetectionBoxes_Recall/AR@100 (small): 0.116933
    INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100 (medium): 0.378204
    I1011 06:28:53.996162 140459371468672 model_lib_v2.py:856] 	+ DetectionBoxes_Recall/AR@100 (medium): 0.378204
    INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100 (large): 0.550939
    I1011 06:28:53.997496 140459371468672 model_lib_v2.py:856] 	+ DetectionBoxes_Recall/AR@100 (large): 0.550939
    INFO:tensorflow:	+ Loss/RPNLoss/localization_loss: 0.244431
    I1011 06:28:53.998564 140459371468672 model_lib_v2.py:856] 	+ Loss/RPNLoss/localization_loss: 0.244431
    INFO:tensorflow:	+ Loss/RPNLoss/objectness_loss: 0.040033
    I1011 06:28:53.999662 140459371468672 model_lib_v2.py:856] 	+ Loss/RPNLoss/objectness_loss: 0.040033
    INFO:tensorflow:	+ Loss/BoxClassifierLoss/localization_loss: 0.206390
    I1011 06:28:54.000716 140459371468672 model_lib_v2.py:856] 	+ Loss/BoxClassifierLoss/localization_loss: 0.206390
    INFO:tensorflow:	+ Loss/BoxClassifierLoss/classification_loss: 0.160537
    I1011 06:28:54.001810 140459371468672 model_lib_v2.py:856] 	+ Loss/BoxClassifierLoss/classification_loss: 0.160537
    INFO:tensorflow:	+ Loss/regularization_loss: 0.000000
    I1011 06:28:54.002920 140459371468672 model_lib_v2.py:856] 	+ Loss/regularization_loss: 0.000000
    INFO:tensorflow:	+ Loss/total_loss: 0.651328
    I1011 06:28:54.003993 140459371468672 model_lib_v2.py:856] 	+ Loss/total_loss: 0.651328
    INFO:tensorflow:Waiting for new checkpoint at ./myoutputmodel
    I1011 06:28:55.316915 140459371468672 checkpoint_utils.py:125] Waiting for new checkpoint at ./myoutputmodel
    Traceback (most recent call last):
      File "/usr/local/lib/python3.6/dist-packages/absl/app.py", line 300, in run
        _run_main(main, args)
      File "/usr/local/lib/python3.6/dist-packages/absl/app.py", line 251, in _run_main
        sys.exit(main(argv))
      File "models/research/object_detection/model_main_tf2.py", line 88, in main
        wait_interval=300, timeout=FLAGS.eval_timeout)
      File "/usr/local/lib/python3.6/dist-packages/object_detection/model_lib_v2.py", line 966, in eval_continuously
        checkpoint_dir, timeout=timeout, min_interval_secs=wait_interval):
      File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/training/checkpoint_utils.py", line 184, in checkpoints_iterator
        checkpoint_dir, checkpoint_path, timeout=timeout)
      File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/training/checkpoint_utils.py", line 132, in wait_for_new_checkpoint
        time.sleep(seconds_to_sleep)
    KeyboardInterrupt
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "models/research/object_detection/model_main_tf2.py", line 113, in <module>
        tf.compat.v1.app.run()
      File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/platform/app.py", line 40, in run
        _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
      File "/usr/local/lib/python3.6/dist-packages/absl/app.py", line 313, in run
        if FLAGS.pdb_post_mortem and sys.stdout.isatty():
      File "/usr/local/lib/python3.6/dist-packages/absl/flags/_flagvalues.py", line 478, in __getattr__
        fl = self._flags()
    KeyboardInterrupt
    ^C


# Evaluation results
    No improvements observed

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.143
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.292
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.119
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.041
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.219
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.397
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.162
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.279
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.293
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.117
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.378
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.551
INFO:tensorflow:Eval metrics at step 250000
```




```python
!cp drive/My\ Drive/myoutputmodel_250k.zip .
!unzip myoutputmodel_250k.zip
```

    Archive:  myoutputmodel_250k.zip
       creating: myoutputmodel/
      inflating: myoutputmodel/ckpt-246.index  
      inflating: myoutputmodel/ckpt-247.data-00000-of-00001  
      inflating: myoutputmodel/ckpt-251.data-00000-of-00001  
      inflating: myoutputmodel/ckpt-249.index  
      inflating: myoutputmodel/ckpt-249.data-00000-of-00001  
      inflating: myoutputmodel/ckpt-248.index  
      inflating: myoutputmodel/checkpoint  
      inflating: myoutputmodel/ckpt-248.data-00000-of-00001  
      inflating: myoutputmodel/ckpt-250.data-00000-of-00001  
       creating: myoutputmodel/train/
      inflating: myoutputmodel/train/events.out.tfevents.1602222489.572f21843685.3333.1504.v2  


```python
!cp drive/My\ Drive/pipeline_file.config .
```

# Export the model for inference


```python
!python models/research/object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path {'./pipeline_file.config'} --trained_checkpoint_dir ./myoutputmodel --output_directory ./myexportedmodel

```

    2020-10-11 06:30:47.410910: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-11 06:30:49.825135: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
    2020-10-11 06:30:49.854697: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:30:49.855423: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
    coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.75GiB deviceMemoryBandwidth: 836.37GiB/s
    2020-10-11 06:30:49.855472: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-11 06:30:49.860160: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2020-10-11 06:30:49.862640: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
    2020-10-11 06:30:49.863103: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
    2020-10-11 06:30:49.868176: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
    2020-10-11 06:30:49.870610: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
    2020-10-11 06:30:49.877470: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    2020-10-11 06:30:49.877620: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:30:49.878337: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:30:49.878929: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
    2020-10-11 06:30:49.879316: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX512F
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2020-10-11 06:30:49.885440: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2000170000 Hz
    2020-10-11 06:30:49.885732: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1edd640 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-10-11 06:30:49.885764: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-10-11 06:30:49.978677: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:30:49.979469: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1edd800 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2020-10-11 06:30:49.979509: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
    2020-10-11 06:30:49.979777: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:30:49.980451: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
    coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.75GiB deviceMemoryBandwidth: 836.37GiB/s
    2020-10-11 06:30:49.980502: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-11 06:30:49.980567: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
    2020-10-11 06:30:49.980593: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
    2020-10-11 06:30:49.980618: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
    2020-10-11 06:30:49.980640: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
    2020-10-11 06:30:49.980662: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
    2020-10-11 06:30:49.980684: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
    2020-10-11 06:30:49.980777: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:30:49.981437: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:30:49.982015: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
    2020-10-11 06:30:49.982068: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
    2020-10-11 06:30:50.702154: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-10-11 06:30:50.702216: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0 
    2020-10-11 06:30:50.702233: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N 
    2020-10-11 06:30:50.702528: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:30:50.703260: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-10-11 06:30:50.703884: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
    2020-10-11 06:30:50.703936: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14756 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0)
    INFO:tensorflow:depth of additional conv before box predictor: 0
    I1011 06:30:55.818547 139864893106048 convolutional_keras_box_predictor.py:154] depth of additional conv before box predictor: 0
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/model_util.py:57: Tensor.experimental_ref (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use ref() instead.
    W1011 06:31:03.033221 139864893106048 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/model_util.py:57: Tensor.experimental_ref (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use ref() instead.
    WARNING:tensorflow:Skipping full serialization of Keras layer <object_detection.meta_architectures.faster_rcnn_meta_arch.FasterRCNNMetaArch object at 0x7f347e5eceb8>, because it is not built.
    W1011 06:31:10.182924 139864893106048 save_impl.py:78] Skipping full serialization of Keras layer <object_detection.meta_architectures.faster_rcnn_meta_arch.FasterRCNNMetaArch object at 0x7f347e5eceb8>, because it is not built.
    2020-10-11 06:31:18.289859: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
    INFO:tensorflow:Assets written to: ./myexportedmodel/saved_model/assets
    I1011 06:31:29.527248 139864893106048 builder_impl.py:775] Assets written to: ./myexportedmodel/saved_model/assets
    INFO:tensorflow:Writing pipeline config file to ./myexportedmodel/pipeline.config
    I1011 06:31:30.003557 139864893106048 config_util.py:254] Writing pipeline config file to ./myexportedmodel/pipeline.config


# Load the model


```python
import tensorflow as tf
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load('./myexportedmodel/saved_model/')
```


```python
from object_detection.utils import label_map_util
label_map_path='./label_map.pbtxt'
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
```


```python
import time
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from object_detection.utils import visualization_utils as viz_utils

%matplotlib inline

cnt = 0
for img in pathlib.Path("bdd100k_data").glob("**/*.jpg"):
  image = Image.open(str(img.resolve()))
  (im_width, im_height) = image.size
  image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

  input_tensor = np.expand_dims(image_np, 0)
  start_time = time.time()
  detections = detect_fn(input_tensor)
  end_time = time.time()
  print("Prediction Time: ", end_time - start_time)
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
  if cnt == 10:
    break
  cnt += 1
```

    Prediction Time:  0.047791242599487305



![png](output_41_1.png)


    Prediction Time:  0.047994136810302734



![png](output_41_3.png)


    Prediction Time:  0.04839324951171875



![png](output_41_5.png)


    Prediction Time:  0.05119013786315918



![png](output_41_7.png)


    Prediction Time:  0.04651761054992676



![png](output_41_9.png)


    Prediction Time:  0.04828381538391113



![png](output_41_11.png)


    Prediction Time:  0.04794764518737793



![png](output_41_13.png)


    Prediction Time:  0.04696989059448242



![png](output_41_15.png)


    Prediction Time:  0.047432661056518555



![png](output_41_17.png)


    Prediction Time:  0.04865860939025879



![png](output_41_19.png)


    Prediction Time:  0.04554605484008789



![png](output_41_21.png)



```python

```
