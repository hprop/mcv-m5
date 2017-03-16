# MCV-M5 : Scene Understanding for Autonomous Vehicles

This is the PreDeeptor (Team 8) repository for the M5 project. Here you can find
the source code, the documents, the deliverables and the instructions
to run the code for each week, and some references that we use for the
project.

## Abstract

Convolutional Neural Networks are a hot topic at this moment. On the other hand, autonomous driving is currently a worry for the society. The current project focuses on implementation and evaluation of deep Convolutional Neural Networks in Object Recognition, Object Detection and Semantic Segmentatation on traffic images.

## Contributors

We are PreDeeptor:

* Ignasi Mas (ignasi.masm@e-campus.uab.cat, Github user: [MrLeylo](https://github.com/MrLeylo))
* Hugo Prol (hugo.prol@e-campus.uab.cat, Github user: [hprop](https://github.com/hprop))
* Jordi Puyoles (jordi.puyoles@e-campus.uab.cat, Github user: [jordi-bird](https://github.com/jordi-bird))

## Documents

* [Overleaf report](https://www.overleaf.com/read/hdtfstjrsqnr)
* [Slides](https://docs.google.com/presentation/d/1wJkGmbYqp0s87yg-msrxecCzeZg3Miuwm85n5_zYxIo/edit?usp=sharing)

## Development

### Week 1. Project presentation

* [VGG network summary](https://docs.google.com/document/d/1zBcWIxjGT02iqhcDFw2RqQj7vJ2ab8TYPH6ApXw5pVU/edit?usp=sharing)
* [ResNet network summary](https://docs.google.com/document/d/1pj0-WEytf4uMvt_VhsHnXWQ4TbrIP5nuCxbp6GX1w8E/edit?usp=sharing)

#### Instructions to run the code

There's no implemented code this week.

### Week 2

#### Abstract

This session focuses on the problem of object classification on the traffic images dataset.

#### Code explained

From the original repository we just worked with the config file and we added 2 models, Resnet and DenseNet.

##### Resnet

We followed the [original paper](https://docs.google.com/document/d/1pj0-WEytf4uMvt_VhsHnXWQ4TbrIP5nuCxbp6GX1w8E/edit?usp=sharing).

##### DenseNet

We followed the [original paper](https://arxiv.org/pdf/1608.06993.pdf). The implementations of [tdeboissiere](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet), [robertomest](https://github.com/robertomest/convnet-study) and [titu1994](https://github.com/titu1994/DenseNet) guided ours. We also added bottleneck and compression algorithms, introduced in the papers.

#### Instructions to run the code

To make a test of a model called *modelX* and save the results in */home/master/folderX*, if you have the datasets in  */home/master/datasets_folder*:

* If you don't have the repository, clone it
* Download its weights on the *Weights* section (file *weights.hdf5* on folder *modelX*) of this Readme and store the file on */home/master/folderX*
* Download its config file on the *Config files* (file *config_modelX.py*) section of this Readme and store it on *mcv-m5/code/config*
* Go to *mcv-m5/code* and run:

`python train.py -c config/config_modelX.py -e ~/folderX -s /data/module5 -l ~/datasets_folder/`

#### Results

The complete description of the experiments and results can be found on the [report](https://www.overleaf.com/read/hdtfstjrsqnr) or on the [presentation](https://docs.google.com/presentation/d/1wJkGmbYqp0s87yg-msrxecCzeZg3Miuwm85n5_zYxIo/edit?usp=sharing), but here we present the obtained accuracy for all of them.

##### VGG tested in TT10K dataset

| Model | Train results | Validation results | Test results |
| :---: | :---: | :---: | :---: |
| VGG16-A | 88.43% | 79.36% | 80.33% |
| VGG16-B | 75.33% | 78.04% | 77.55% |
| VGG16-C | 96.30% | 92.36% | 94.02% |
| VGG16-D | 95.05% | 90.55% | 91.91% |
| VGG16-A+ | 95.87% | 92.61% | 92.93% |
| VGG16-A++ | 99.56% | 95.54% | 96.69% |

##### VGG tested in BelgiumTSC dataset

| Model | Train results | Validation results | Test results |
| :---: | :---: | :---: | :---: |
| VGG16-E | 81.25% | 76.55% | 75.67% |

##### VGG tested in KITTI dataset

| Model | Train results | Validation results | Test results |
| :---: | :---: | :---: | :---: |
| VGG16-F | 94.55% | 93.67% | 93.67% |

##### Additional architectures tested in  dataset

| Model | Train results | Validation results | Test results |
| :---: | :---: | :---: | :---: |
| ResNet-A | 99.89% | 95.72% | 92.92% |
| ResNet-B | 75.84% | 75.15% | (\*) |
| ResNet-B+ | 99.55% | 83.75% | (\*) |
| DenseNet-A | 99.89% | 91.83% | 92.42% |
| DenseNet-B | 97.86% | 88.03% | 84.92% |

(*)*Due to technical issues we don't have the results for that models*

#### Config files

On the folder below, you can download the config files to test each model.

[Mirror](https://drive.google.com/drive/folders/0BzPpFJ8eI8VSTGRGb3ZrZ1RBajg)

#### Weights

On the folder below, you can access to the folder which stores the weights of each model.

[Mirror](https://drive.google.com/drive/folders/0BzPpFJ8eI8VSUm5JQzBjMDJtaGc)

### Week 3

#### Instructions to run the code

### Week 4

#### Instructions to run the code

### Week 5

#### Instructions to run the code

### Week 6

#### Instructions to run the code

## References

* Simonyan, K., Zisserman, A. [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf). CoRR 2014.
* He, K., Zhang, X., Ren, S. & Sun, J. [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf). CoRR 2015.
* Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten[Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf), 2016
* liuzhuang13, [Code for Densely Connected Convolutional Networks (DenseNets)](https://github.com/liuzhuang13/DenseNet)
