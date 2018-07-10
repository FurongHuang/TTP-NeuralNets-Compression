# Tensorized Spectrum Preserving Compression of Neural Networks
We implement an efficient mechanism for compressing large networks by **tensorizing** network layers: i.e. mapping layers on to high-order matrices, for which we introduce new tensor decomposition methods. 

To Cite: This software is an implementation of the following paper 

Su, Jiahao, Jingling Li, Bobby Bhattacharjee, and Furong Huang. "Tensorized Spectrum Preserving Compression for Neural Networks." arXiv preprint arXiv:1805.10352 (2018).

# Overview
Our program is based on top of [Tensorflow's official framework for resnet](https://github.com/tensorflow/models/tree/master/official/resnet), which can be applied to resnet with various sizes. We focused our experiments on the CIFAR10 dataset and the ImageNet (2012) dataset. We break the compression process into 3 steps: **Phase0**, **Phase1**, and **Phase2**.

## Phase0 (Tensor decomposition): 
Apply a given (high-order) tensor decomposition to each layer's weight matrix (or weight tensor, e.g. the weight of a convolutional layer) of a pretrained model, and build a corresponding neural network framework use the decomposed components of the original weight matrix for each layer.

## Phase1 (Sequential training):
We conduct fine tuning on the network model obtained from phase0 by dividing the network into several blocks, and start training on the 1st, 2nd, ... blocks sequentially. We will finish the training for the 1st to the the kth block before we start fine tuning on the (k+1)th block. This sequential training step uses the pretrained model as the reference network in fine tuning (aka. the loss function is the difference between the output of our model and that of the pretrained model).

## Phase2 (End-to-end fine tuning):
Conduct end-to-end training (normal loss function like cross entropy) on the network model obtained from phase1.

# Set up
## Dependencies
To sucessfully run the scripts, besides usual packages for Tensorflow, you also need to install Tensorly. The installation is quite simple and you can use pip as what is mentioned in the [instructions](http://tensorly.org/stable/installation.html)

## Data 
You can follow the instructions on [this link](https://github.com/tensorflow/models/tree/master/official/resnet) to download the CIFAR-10 dataset and the ImageNet dataset.

## Pretrained models
As it is quite fast to train CIFAR-10 end to end, the pretrained model we used for cifar10 is trained using the default settings in the [official resnet framework] (https://github.com/tensorflow/models/tree/master/official/resnet). And the the pretrained model for ImageNet is the "ResNet-50 v2" one in the [official resnet framework](https://github.com/tensorflow/models/tree/master/official/resnet).

# How to run
We implemented our own convolutional layer and fully connected layer, which can support compression using 7 different tensor decomposition techniques with specified compression rates. You can find the implementation under the tensornet/ folder.

We also prepared automated scripts (under the scripts/ folder) as a demonstration of how we applied the customized conduct convolutional layer and fully connected layer on ResNet used for the CIFAR-10 dataset and the ImageNet dataset. Feel free to add additional flags in the scripts to have richer features in training (e.g. --multi_gpu)

The order of the parameters corresponding to the following macros in the scripts:

| Order  | Macro Name | Meaning  |
| ---- |:----------:|:------------:|
| 1      | METHOD | the tensor decomposition method used for neural network compression |
| 2      | RATE      |  the compression rate |
| 3      | WORK_DIR  | the directory where this github repository lies |
| 4      | DATA_DIR | the directory where the training data lies |
| 5      | PRETRAINED_MODEL      |  the directory where the pretrained model lies |
| 6      | OUTPUT_DIR  | the directory where the output should lie |
| 7      | BATCH_SIZE      |  the batch size used for fine-tuning (phase 1 and phase 2) |
| 8      | EPOCHS  | the number of train epoches used for fine-tuning (phase 1 and phase 2)|

For example, 
- **bash cifar10_script.sh 'cp' 0.1 '/TTP-NeuralNets-Compression' '/Data/cifar10_data' '/models/cifar10_model' '/tensorized_models/cifar10' 128 100**

means conduct the three phase compression process using CP-decomposition with 10% compression rate on resnet model used for CIFAR-10, where the work directory is '/TTP-NeuralNets-Compression', the CIFAR-10 dataset is stored under '/Data/cifar10_data', the pretrained resnet model is stored under '/models/cifar10_model', and the expected ouput directory is '/tensorized_models/cifar10'. Here, the batch_size is 128 and number of training epoches is 100.

- **bash imagenet_script.sh 'tk' 0.05 '/TTP-NeuralNets-Compression' '/Data/ImageNet2012' '/models/imagenet_resnet50' '/tensorized_models/imagenet' 256 50**

means conduct the three phase compression process using Tucker-decomposition with 5% compression rate on resnet model used for ImageNet, where the work directory is '/TTP-NeuralNets-Compression', the ImageNet dataset is stored under '/Data/ImageNet2012', the pretrained resnet model is stored under '/models/imagenet_resnet50', and the expected ouput directory is '/tensorized_models/imagenet'. Here, the batch_size is 256 and number of training epoches is 50.
