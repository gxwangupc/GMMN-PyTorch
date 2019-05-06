# Generative Moment Matching Networks(GMMN)
-------------------------------------------------
## Introduction
 * The code is widely adapted from two nice repositories:<br>
<https://github.com/Abhipanda4/GMMN-Pytorch> <br>
<https://github.com/siddharth-agrawal/Generative-Moment-Matching-Networks>. <br>
 * I failed to implement the greedy layer-wise pretraining and fine-tuning scheme mentioned in the paper as I have not fully understood it.

## Environment & Requirements
* CentOS Linux release 7.2.1511 (Core)<br>
* python 3.6.5<br>
* torch  1.0.0<br>
* torchvision<br>
* argparse<br>
* pickle

## Architecture
![](https://github.com/gxwangupc/GMMN-PyTorch/blob/master/architecture.png)
## Usage
### 1. Train the networks using MNIST:<br>
    python train.py
Two folders will be created, i.e., data & models. As their names imply, they are used to store data and trained models, repectively. 
### 2. Visualize the outputs of the gmmn:  <br>
    python Visualize.py --visualize gmmn
### 3. Visualize the outputs of the autoencoder:  
    python Visualize.py --visualize autoencoder

## References 
<https://github.com/Abhipanda4/GMMN-Pytorch><br>
<https://github.com/siddharth-agrawal/Generative-Moment-Matching-Networks>
