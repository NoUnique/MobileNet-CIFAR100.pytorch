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
|SEMobileNetV2|76.64|(77.9)|309.84|16.05|4|train-semobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm2.0_rn8_s1,t1_c16_n1_s1,t4_c24_n2_s1,t4_c40_n2_s2,t4_c80_n3_s2,t4_c112_n3_s1,t4_c192_n4_s2,t4_c320_n1_s1|
|SEMobileNetV2|75.92|(77.6)|118.90|7.16|4|train-semobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm1.0_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s1,t6_c40_n2_s2,t6_c80_n3_s2,t6_c112_n3_s1,t6_c192_n4_s2,t6_c320_n1_s1|
|SEMobileNetV2|75.61|(77.1)|183.84|9.13|4|train-semobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm1.5_rn8_s1,t1_c16_n1_s1,t4_c24_n2_s1,t4_c40_n2_s2,t4_c80_n3_s2,t4_c112_n3_s1,t4_c192_n4_s2,t4_c320_n1_s1|
|SEMobileNetV2|75.10|(76.6)|94.29|4.61|4|train-semobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm1.0_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s1,t6_c32_n3_s2,t6_c64_n4_s2,t6_c96_n3_s1,t6_c160_n3_s2,t6_c320_n1_s1|
|SEMobileNetV2|74.71|(76.2)|81.57|4.32|4|train-semobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm0.75_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s1,t6_c40_n2_s2,t6_c80_n3_s2,t6_c112_n3_s1,t6_c192_n4_s2,t6_c320_n1_s1|
|SEMobileNetV2|74.70|(76.4)|81.39|4.12|4|train-semobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm1.0_rn8_s1,t1_c16_n1_s1,t4_c24_n2_s1,t4_c40_n2_s2,t4_c80_n3_s2,t4_c112_n3_s1,t4_c192_n4_s2,t4_c320_n1_s1|
|SEMobileNetV2|74.63|(75.9)|81.83|4.61|4|train-semobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm1.0_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s2,t6_c32_n3_s1,t6_c64_n4_s2,t6_c96_n3_s1,t6_c160_n3_s2,t6_c320_n1_s1|
|SEMobileNetV2|74.52|(76.3)|64.11|2.76|4|train-semobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm0.75_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s1,t6_c32_n3_s2,t6_c64_n4_s2,t6_c96_n3_s1,t6_c160_n3_s2,t6_c320_n1_s1|
|SEMobileNetV2|73.44|(75.2)|56.13|2.54|4|train-semobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm0.75_rn8_s1,t1_c16_n1_s1,t4_c24_n2_s1,t4_c40_n2_s2,t4_c80_n3_s2,t4_c112_n3_s1,t4_c192_n4_s2,t4_c320_n1_s1|
|SEMobileNetV2|73.09|(74.7)|44.70|1.71|4|train-semobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm0.75_rn8_s1,t1_c16_n1_s1,t4_c24_n2_s1,t4_c32_n3_s2,t4_c64_n4_s2,t4_c96_n3_s1,t4_c160_n3_s2,t4_c320_n1_s1|
|MobileNetV2|74.34|(75.8)|91.37|2.35|4|train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm1.0_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s1,t6_c32_n3_s2,t6_c64_n4_s2,t6_c96_n3_s1,t6_c160_n3_s2,t6_c320_n1_s1|
|MobileNetV2|73.84|(75.7)|79.10|2.35|4|train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm1.0_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s2,t6_c32_n3_s1,t6_c64_n4_s2,t6_c96_n3_s1,t6_c160_n3_s2,t6_c320_n1_s1|
|MobileNetV2|73.51|(75.1)|62.27|1.48|4|train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm0.75_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s1,t6_c32_n3_s2,t6_c64_n4_s2,t6_c96_n3_s1,t6_c160_n3_s2,t6_c320_n1_s1|
|MobileNetV2|72.04|(74.0)|43.74|1.14|4|train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm0.75_rn8_s1,t1_c16_n1_s1,t4_c24_n2_s1,t4_c32_n3_s2,t4_c64_n4_s2,t4_c96_n3_s1,t4_c160_n3_s2,t4_c320_n1_s1|
|MobileNetV2|70.35|(72.8)|29.92|0.82|4|train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm0.5_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s1,t6_c32_n3_s2,t6_c64_n4_s2,t6_c96_n3_s1,t6_c160_n3_s2,t6_c320_n1_s1|
|MobileNetV2|68.15|(70.2)|25.67|2.35|4|train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm1.0_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s2,t6_c32_n3_s2,t6_c64_n4_s2,t6_c96_n3_s1,t6_c160_n3_s2,t6_c320_n1_s1|
|MobileNetV2 (baseline)|56.61|(59.6)|6.51|2.35|4|train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm1.0_rn8_s2,t1_c16_n1_s1,t6_c24_n2_s2,t6_c32_n3_s2,t6_c64_n4_s2,t6_c96_n3_s1,t6_c160_n3_s2,t6_c320_n1_s1|

### Pretrained Models 
|network|top1-acc|MACs(M)|params(M)|checkpoint|ngpus|config|block_args|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SEMobileNetV2|76.64|309.84|16.05|[TBA]()|4|train-semobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm2.0_rn8_s1,t1_c16_n1_s1,t4_c24_n2_s1,t4_c40_n2_s2,t4_c80_n3_s2,t4_c112_n3_s1,t4_c192_n4_s2,t4_c320_n1_s1|
|SEMobileNetV2|75.92|118.90|7.16|[TBA]()|4|train-semobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm1.0_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s1,t6_c40_n2_s2,t6_c80_n3_s2,t6_c112_n3_s1,t6_c192_n4_s2,t6_c320_n1_s1|
|SEMobileNetV2|75.10|94.29|4.61|[TBA]()|4|train-semobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm1.0_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s1,t6_c32_n3_s2,t6_c64_n4_s2,t6_c96_n3_s1,t6_c160_n3_s2,t6_c320_n1_s1|
|MobileNetV2|74.34|91.37|2.35|[TBA]()|4|train-mobilenetv2-cifar100-b128-e200-w5-cosine-wd0.0001-lr0.1.conf|wm1.0_rn8_s1,t1_c16_n1_s1,t6_c24_n2_s1,t6_c32_n3_s2,t6_c64_n4_s2,t6_c96_n3_s1,t6_c160_n3_s2,t6_c320_n1_s1|
  

## Contact
Taehwan Yoo (kofmap@gmail.com)
