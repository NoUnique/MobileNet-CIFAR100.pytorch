# A PyTorch implementation for MobileNet on CIFAR100
This repository is for training MobileNets on CIFAR100 using pytorch 
and [Docker-based DL development environment](https://github.com/NoUnique/devenv.docker)

* Model codes are based on [torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py).
* Block argument parser is based on [EfficientNet](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py)

## How to Use
You have to clone this repository recursively
```
git clone --recursive https://github.com/NoUnique/MobileNet-CIFAR100.pytorch.git
```
To build Docker image
```
./docker/compose -b
```
To run docker container for DL development
```
./docker/compose -r
```
To attach the container
```
./docker/compose -s
```

<br>

_**You must run code below in the container**_

<br>

To train MobileNetV2 on CIFAR-100 dataset with a single-GPU:
```
CUDA_VISIBLE_DEVICES=0 python train.py --flagfile configs/train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf
```
To train MobileNetV2 on CIFAR-100 dataset with 4 GPUs:
```
horovodrun -np 4 python train.py --flagfile configs/train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf
```
To train MobileNetV2 with custom block_args on CIFAR-100 dataset with 4 GPUs:
```
horovodrun -np 4 python train.py --flagfile configs/train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf --BLOCK_ARGS=wm1.0_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s1,t6_c32_n3_s2,t6_c64_n4_s2,t6_c96_n3_s1,t6_c160_n3_s2,t6_c320_n1_s1
```
To test trained model(last checkpoint) on CIFAR-100 dataset with a single-GPU:
```
CUDA_VISIBLE_DEVICES=0 python test.py --flagfile configs/train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf
```
To test trained model(specific checkpoint) on CIFAR-100 dataset with a single-GPU:
```
CUDA_VISIBLE_DEVICES=0 python test.py --flagfile configs/train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf --PRETRAINED_CHECKPOINT_PATH=checkpoints/train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1/checkpoint-200.pth.tar
```
If you test other block_args setting, you have to specify 'BLOCK_ARGS' flag
- default: MobileNetV2 with stride 1(stem), 1, 1, 2, 2, 1, 2, 1, other args are same to paper
```
CUDA_VISIBLE_DEVICES=0 python test.py --flagfile configs/train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf --BLOCK_ARGS=wm1.0_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s1,t6_c32_n3_s2,t6_c64_n4_s2,t6_c96_n3_s1,t6_c160_n3_s2,t6_c320_n1_s1
```

### Tracking training progress with TensorBoard
To run tensorboard service to 6006 port
```
./docker/compose --tensorboard
```

### Experimental Results
- validation accuracy is calculated during training via horovod(it may not correct)

|network|top1-acc|val-acc|MACs(M)|params(M)|ngpus|config|block_args|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|MobileNetV2 |74.34|(75.8)|91.37|2.35|4|train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|['wm1.0_rn8_s1', 't1_c16_n1_s1', 't6_c24_n2_s1', 't6_c32_n3_s2', 't6_c64_n4_s2', 't6_c96_n3_s1', 't6_c160_n3_s2', 't6_c320_n1_s1']|
|MobileNetV2 |68.15|(70.2)|25.67|2.35|4|train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|['wm1.0_rn8_s1', 't1_c16_n1_s1', 't6_c24_n2_s2', 't6_c32_n3_s2', 't6_c64_n4_s2', 't6_c96_n3_s1', 't6_c160_n3_s2', 't6_c320_n1_s1']|
|MobileNetV2 (baseline)|56.61|(59.6)|6.51|2.35|4|train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|['wm1.0_rn8_s2', 't1_c16_n1_s1', 't6_c24_n2_s2', 't6_c32_n3_s2', 't6_c64_n4_s2', 't6_c96_n3_s1', 't6_c160_n3_s2', 't6_c320_n1_s1']|

### Pretrained Models 
|network|top1-acc|MACs(M)|params(M)|checkpoint|ngpus|config|block_args|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|MobileNetV2 (best)|74.34|91.37|2.35|[TBA]()|4|train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|['wm1.0_rn8_s1', 't1_c16_n1_s1', 't6_c24_n2_s1', 't6_c32_n3_s2', 't6_c64_n4_s2', 't6_c96_n3_s1', 't6_c160_n3_s2', 't6_c320_n1_s1']|
  

## Contact
Taehwan Yoo (kofmap@gmail.com)
