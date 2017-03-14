# MCV-M5 : Scene Understanding for Autonomous Vehicles

This is the PreDeeptor (Team 8) repository for the M5 project. Here you can find
the source code, the documents, the deliverables and the instructions
to run the code for each week, and some references that we use for the
project.

## Abstract

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



#### Code explained

From the original repository we just worked with the config file and we added 2 models, Resnet and DenseNet.

##### Resnet

We followed the [original paper](https://docs.google.com/document/d/1pj0-WEytf4uMvt_VhsHnXWQ4TbrIP5nuCxbp6GX1w8E/edit?usp=sharing).

##### DenseNet

We followed the [original paper](https://arxiv.org/pdf/1608.06993.pdf). The implementations of [tdeboissiere](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet), [robertomest](https://github.com/robertomest/convnet-study) and [titu1994](https://github.com/titu1994/DenseNet) guided ours. We also added bottleneck and compression algorithms, introduced in the papers.

#### Instructions to run the code

#### Results


| Model | Train results | Validation results | Test results | 
| :---: | :---: | :---: | :---: |
| VGG16-A |  |  |  |
| VGG16-B |  |  |  |
| VGG16-C |  |  |  |
| VGG16-D |  |  |  |
| VGG16-E |  |  |  |
| VGG16-A+ |  |  |  |
| VGG16-F |  |  |  |
| Resnet-A |  |  |  |
| Resnet-B |  |  |  |
| DenseNet-AB |  |  |  |
| Denseet-B |  |  |  |


#### Weights

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
