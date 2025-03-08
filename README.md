# Say No to Freeloader: Protecting Intellectual Property of Your Deep Model

Code release for "Say No to Freeloader: Protecting Intellectual Property of Your Deep Model" (PAMI 2024)

## Paper

<div align=center><img src="https://github.com/LyWang12/CUPI-Domain/blob/main/Figure/figure.png" width="100%"></div>

[Say No to Freeloader: Protecting Intellectual Property of Your Deep Model](https://arxiv.org/abs/2408.13161) 
(PAMI 2024)

We propose a Compact Un-transferable Pyramid Isolation Domain (CUPI-Domain) which serves as a barrier against illegal transfers from authorized to unauthorized domains, to protect the intellectual property (IP) of AI models.

## Prerequisites
The code is implemented with **CUDA 11.4**, **Python 3.8.5** and **Pytorch 1.8.0**.

## Datasets

### MNIST
MNIST dataset can be found [here](http://yann.lecun.com/exdb/mnist/).

### USPS
USPS dataset can be found [here](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2).

### SVHN
SVHN dataset can be found [here](http://ufldl.stanford.edu/housenumbers/train_32x32.mat).

### MNIST-M
MNIST-M dataset can be found [here](https://arxiv.org/pdf/1505.07818v4.pdf).

### CIFAR-10
CIFAR-10 dataset can be found [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

### STL-10
CIFAR-10 dataset can be found [here](https://opendatalab.com/STL-10).

### VisDA 2017
VisDA 2017 dataset can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public).

### Office-Home
Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

### DomainNet
DomainNet dataset can be found [here](http://ai.bu.edu/DomainNet/).


## Running the code

Target-Specified CUPI-Domain
```
python train_ts_dight.py
```

Ownership Verification by CUPI-Domain
```
python train_owner_dight.py
```

Target-free CUPI-Domain
```
python train_tf_dight.py
```

Applicability Authorization by CUPI-Domain
```
python train_author_dight.py
```

## Citation
If you find this code useful for your research, please cite our [paper](https://arxiv.org/abs/2408.13161):
```
@article{wang2024say,
  title={Say No to Freeloader: Protecting Intellectual Property of Your Deep Model},
  author={Wang, Lianyu and Wang, Meng and Fu, Huazhu and Zhang, Daoqaing},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgements
Some codes are adapted from [NTL](https://github.com/conditionWang/NTL) and 
[SWIN-Transformer](https://github.com/microsoft/Swin-Transformer). We thank them for their excellent projects.

## Contact
If you have any problem about our code, feel free to contact
- lywang12@126.com
- wangmeng9218@126.com

or describe your problem in Issues.
