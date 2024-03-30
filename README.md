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
|            | AP75 | AP50 | AP25 | Models |
| ---------- | ---- | ---- | ---- | ------ |
| Meccano    | 14.4 | 28.8 | 36.2 |    [meccano](https://pan.baidu.com/s/1jNLnWiYZqqeYvJk3H7GnPw?pwd=KAD0)    |
| 100DOH     | 31.2 | 53.9 | 58.9 |    [100DOH](https://pan.baidu.com/s/1jNLnWiYZqqeYvJk3H7GnPw?pwd=KAD0)     |
|            | AP   | AP50 | AP75 | Models |
| ego4d-swin | 40.5 | 60.6 | 41.9 |    [ego4d-swin](https://pan.baidu.com/s/1jNLnWiYZqqeYvJk3H7GnPw?pwd=KAD0)     |
| ego4d-r50  | 31.4 | 34.6 | 28.9 |     [ego4d-r50](https://pan.baidu.com/s/1jNLnWiYZqqeYvJk3H7GnPw?pwd=KAD0)      |
| epic-swin  | 35.2 | 44.1 | 32.5 |  [epic-swin](https://pan.baidu.com/s/1jNLnWiYZqqeYvJk3H7GnPw?pwd=KAD0)         |
| epic-r50   | 30.2 | 30.1 | 22.5 |     [epic-r50](https://pan.baidu.com/s/1jNLnWiYZqqeYvJk3H7GnPw?pwd=KAD0)   |



## Reference
If you find our work useful in your research, please consider citing our paper:
```
# TODO
```

## Acknowledgements
We would like to thank the authors of [mmdetection](github.com/open-mmlab/mmdetection) for providing the codebase for object detection.




