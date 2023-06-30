# A baseline framework for federated-learning

This is a baseline framework for federated learning.

It supports various datasets and optimization strategies.

---
## Index
1. [Install requirements](#install-requirements)
1. [Catalog](#catalog)
1. [Training options](#training-options)

---

## [Install requirements](#index)

To install requirements, run the command below:

```
pip install -r requirements.txt
```

## [Catalog](#index)
1. [Datasets](#datasets)
1. [Models](#models)
1. [Federated learning strategies](#federated-learning-strategies)

### Datasets

- [x] MNIST: 10-digit dataset.
- [x] Fashion-MNIST: clothing dataset with 10 classes.
- [x] CIFAR-10, CIFAR-100: image dataset with 10 and 100 classes, respectively.
- [x] FEMNIST: large version of MNIST for federated learning benchmarking.
This project contains a small version of FEMNIST with 520 clients.

### Models

- [x] Simple CNN models: used in various FL papers.
- [x] ResNets: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 are supported.

### Federated learning strategies

- [x] FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data [[`arXiv`](https://arxiv.org/abs/1602.05629)]
- [x] SCAFFOLD: Stochastic Controlled Averaging for Federated Learning [[`arXiv`](https://arxiv.org/abs/1910.06378)]
- [x] FedDyn: Federated Learning Based on Dynamic Regularization [[`arXiv`](https://arxiv.org/abs/2111.04263)]

## [Training options](#index)
1. [Federated learning arguments](#federated-learning-arguments)
1. [Model and dataset arguments](#model-and-dataset-arguments)
1. [Optimizing arguments](#optimizing-arguments)
1. [Misc](#misc)

### Federated learning arguments
- epochs (default=300): total rounds of federated learning training
- n_clients (default=100): number of clients participating in federated learning
- frac (default=0.1): fraction of clients participating in each round. 1.0 for full participation
- local_ep (default=5): number of local epochs for each client
- local_bs (default=100): local batch size
- test_bs (default=128): test batch size
- lr (default=0.01): learning rate
- lr_decay (default=0.9): learning rate decay
- lr_decay_step_size (default=500): step size to decay learning rate

### Model and dataset arguments
- model (default='cnn'): model used for training (options: 'mlp', 'cnn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
- dataset (default='mnist'): name of dataset (options: 'mnist', 'fashion-mnist', 'femnist', 'cifar10', 'cifar100')
- iid (action='store_true'): whether i.i.d or notc (default: non-iid)
- spc (action='store_true'): whether spc or not (default: Dirichlet distribution)
- beta (default=0.2): beta for Dirichlet distribution
- n_classes (default=10): number of classes. Automatically modified in code
- n_channels (default=1): number of channels. Automatically modified in code

### Optimizing arguments
- optimizer (default='sgd'): Optimizer (options: SGD, Adam)
- momentum (default=0.0): SGD momentum
- fed_strategy (default='fedavg'): optimization scheme (options: 'fedavg', 'scaffold', 'feddyn')
- alpha (default=1.0): alpha for feddyn

### Misc
- n_gpu (default=4): number of GPUs to use
- n_procs (default=1): number of processes per processor
- seed (default=0): random seed
- no_record (action='store_true'): whether to record or not (default: record)


