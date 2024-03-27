# Active Object Detection with Knowledge Aggregation and Distillation
> This repository contains the code for the paper "Active Object Detection with Knowledge Aggregation and Distillation" accepted at CVPR 2024.


## Requirements
- Python>=3.10.9
- torch>=1.13.1
- torchvision>=0.14.1
- mmcv>= 2.1.0
- mmdet>=3.3.0
- mmengine >=0.10.3
- timm>=0.6.13
- loguru
- `requirements.txt` file is provided for easy installation of the required packages.

## Datasets
We evaluate our method on the following datasets:
- MECCANO
- 100DOH
- EPIC
- Ego4D

## Training
To train the teacher model, run the following command:
```bash
# for example [meccano]:
bash tools/dist_train.sh configs/active_object/meccano.py [num_gpus]
```

## Evaluation
To evaluate the student model, run the following command:
```bash
# for example [meccano]:
bash tools/dist_test.sh configs/active_object/meccano.py [path_to_checkpoint] [num_gpus]
```

## Checkpoints
Checkpoints for  models are provided in the `Release`  of the repo.

## Reference
If you find our work useful in your research, please consider citing our paper:
```
# TODO
```

## Acknowledgements
We would like to thank the authors of [mmdetection](github.com/open-mmlab/mmdetection) for providing the codebase for object detection.




