# Image Tagging using CNN
Project of course Deep learning and its applications. SOICT - HUST

## Description

**Included models:**
1. [BaseResNet50V2](src/model/base_model.py): Using pretrained Resnet50V2
2. [ResNet50](src/model/resnet.py): Implementing model based on ResNet architecture
3. [AttentionVGG](src/model/acnn_v1.py): Implementing model based on Visual Attention Network
4. [AttentionResNet](src/model/acnn_v2.py): Applying attention mechanism into ResNet model

## Run project
1. Run `$ ./run_build.sh` or `$ bash run_build.sh`
2. Edit project's configuration in [config.yml](config.yml)
3. For training: `$ python train.py`
4. For testing: `$ python test.py`
