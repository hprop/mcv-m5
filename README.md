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

### Week 2. Object recognition

#### Code explained

From the original repository we just worked with the config file and we added 2 models, Resnet and DenseNet.

##### Resnet

We followed the [original paper](https://docs.google.com/document/d/1pj0-WEytf4uMvt_VhsHnXWQ4TbrIP5nuCxbp6GX1w8E/edit?usp=sharing).

##### DenseNet

We followed the [original paper](https://arxiv.org/pdf/1608.06993.pdf). The implementations of [tdeboissiere](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet), [robertomest](https://github.com/robertomest/convnet-study) and [titu1994](https://github.com/titu1994/DenseNet) guided ours. We also added bottleneck and compression algorithms, introduced in the papers.

#### Achievements

* Finetune and test the VGG16 model over TT100K dataset: with cropped images.
* Finetune and test the VGG16 model over TT100K dataset: with entire images.
* Repeat those experiments training from scratch.
* Train the VGG16 model (from scratch with entire images) with transfer learning over the BTS dataset.
* Train and test the VGG16 model over Kitti dataset.
* Accelerate the previous training: downsample the images.
* Train from scratch a ResNet model over TT100K dataset.
* Finetune a ResNet model over TT100K dataset.
* Train from scratch a DenseNet model over TT100K dataset.
* Handle with the amount of parameters in DenseNet: reduce the number of layers and filters and the growth rate.
* Accelerate the previous training and its learning process: use bottleneck and compression in DenseNet and increase the learning rate.
* Perform the previous test wih dropout.

#### Instructions to run the code

To make a test of a model called *modelX* and save the results in */home/master/folderX*, if you have the datasets in  */home/master/datasets_folder*:

* If you don't have the repository, clone it
* Download its weights on the *Weights* section (file *weights.hdf5* on folder *modelX*) of this Readme and store the file on */home/master/folderX*
* Download its config file on the *Config files* (file *config_modelX.py*) section of this Readme and store it on *mcv-m5/code/config*
* Go to *mcv-m5/code* and run:

`python train.py -c config/config_modelX.py -e ~/folderX -s /data/module5 -l ~/datasets_folder/`

#### Weights

On the folder below, you can access to the folder which stores the weights of each model.

[Mirror](https://drive.google.com/drive/folders/0BzPpFJ8eI8VSUm5JQzBjMDJtaGc)

### Week 3 & 4. Object detection

#### Achievements

* Train the given YOLO network with its default configuration.
* Inspect the TT100k dataset limitations: differences in train and test sets.
* Confirm the (expected) effect of those limitations: gap between train set and the rest.
* Evaluate the train results: f-score.
* Train the Tiny-YOLO network: less time per frame (almost the half) but performance better in YOLO.
* Inspect the Udacity: differences in conditions in train and test.
* Train YOLO in Udacity dataset: high effect of the limitation above.
* Boost YOLO over TT100k dataset: preprocessing techniques (samplewise normalization, global contrast normalization).
* Boost previous training: increasing the initial learning rate but with an early decay.
* Read papers for dalternative architectures and pick: SSD.
* Implement this network, train, test and evaluate results.

#### Code explained

##### YOLO

We modified the global contrast normalization (GCN) provided in the
framework since it appears broken due to the introduction of a mask
array to handle void labels (for semantic segmentation). GCN was one
of the preprocessing stages used in our experiments with the YOLO
architecture.

Contributions were also done in the *eval_detection_fscore* script to
add the preprocessing stages used (samplewise center, std
normalization, GCN).

##### SSD

Our implementation is based on the code from
the [rykov8's repository](https://github.com/rykov8/ssd_keras).

Beyond some modifications to adapt the input and output bounding box
formats to those used in our framework, our major contribution was to
decouple the base model from the priors declaration and the
construction of the prediction layers. Thus we are able to build easily
new SSD topologies with the `build_ssd()` function (see *models/ssd.py*).

We plan to add in further contributions (out of assignment) a SSD
architecture with a resnet base model.

#####Modifications on the framework

* Global contrast normalization in code/tools/data_loader.py to be computed over all the image.


#### Instructions to run the code

To make a test of the experiment corresponding to the config file *experimentX* in the *code/config* folder on the repository and save the results in */home/master/folderX*, if you have the datasets in  */home/master/datasets_folder*:

* If you don't have the repository, clone it
* Download its weights on the *Weights* section (file *weights.hdf5* on folder *experimentX*) of this Readme and store the file on */home/master/folderX*
* Go to *mcv-m5/code* and run:

`python train.py -c config/experimentX.py -e ~/folderX -s /data/module5 -l ~/datasets_folder/`

To evaluate the f-score of the model generated by the previous experiment:

* Go to *mcv-m5/code* and run:

`python eval_detection_fscore.py ~/folderX/weights.hdf ~/datasets_folder`

#### Weights

On the folder below, you can access to the folder which stores the weights of each model.

[Mirror](https://drive.google.com/drive/folders/0BzPpFJ8eI8VSYkYzVkM5RlRNY3M?usp=sharing)

### Week 5 & 6. Object segmentation

#### Code explained

From the original repository we made some modifications on the framework, worked with the config files and added one model, Tiramisu.

##### Tiramisu

We followed the [original paper](https://arxiv.org/pdf/1611.09326.pdf). We also based our model on [SimJeg's implementation](https://github.com/SimJeg/FC-DenseNet). This was implemented in Lasagne, we implemented it in Keras.

To solve some missmatches we do Zero Padding after deconvolutional layers (to concatenate with the skip connections). Bottleneck and compression algorithms are implemented. We also implemented eval_dataset.py

#####Modifications on the framework

* Custom Cropping2D layer in layers/outlayers.py (it handles symbolic input shapes as in keras version 2).


#### Achievements

* Train and test the FCN8 model over Camvid dataset.
* Boost FCN8 over Camvid dataset: finetuning.
* Boost FCN8 over Camvid dataset: finetuning with data augmentation.
* Evaluate other datasets with their class distribution, image properties, dataset size or other factors. Pick one for further experiments with FCN8: Synthia.
* Boost FCN8 over Synthia dataset: finetuning.
* Read papers and select another segmentation architecture to train in Camvid: Tiramisu.
* Boost Tiramisu over Camvid dataset: finetuning with data augmentation and bilinear initialization on deconvolutional layers.
* Handle with Tiramisu high dimensionality on training data: batch size limit.

#### Instructions to run the code

To make a test of the experiment corresponding to the config file *experimentX* in the *code/config* folder on the repository and save the results in */home/master/folderX*, if you have the datasets in  */home/master/datasets_folder*:

* If you don't have the repository, clone it
* Download its weights on the *Weights* section (file *weights.hdf5* on folder *experimentX*) of this Readme and store the file on */home/master/folderX*
* Go to *mcv-m5/code* and run:

`python train.py -c config/experimentX.py -e ~/folderX -s /data/module5 -l ~/datasets_folder/`

#### Weights

On the folder below, you can access to the folder which stores the weights of each model.

[Mirror](https://drive.google.com/open?id=0BzPpFJ8eI8VSQ2NVd2N4YXNPaHM)

## References

* Simonyan, K., Zisserman, A. [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf). CoRR 2014.
* He, K., Zhang, X., Ren, S. & Sun, J. [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf). CoRR 2015.
* Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten[Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf), 2016
* liuzhuang13, [Code for Densely Connected Convolutional Networks (DenseNets)](https://github.com/liuzhuang13/DenseNet)
* [TSingHua-TenCent 100K dataset](http://cg.cs.tsinghua.edu.cn/traffic-sign/)
* [KITTI Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php)
* [KUL Belgium Traffic Signs dataset](http://btsd.ethz.ch/shareddata/index.html)
* [Udacity Dataset](https://github.com/udacity/self-driving-car/tree/master/annotations)
* [ia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Fei-Fei Li. [Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255. IEEE Computer Society, 2009](http://image-net.org/)
* Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. [Rich feature hierarchies for accurate object detection and semantic segmentation. In Computer Vision and Pattern Recognition](https://arxiv.org/pdf/1311.2524.pdf)
* W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg. [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)
* J. Redmon, S. Divvala, R. Girshick, and A. Farhadi.[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
* J. Redmon and A. Farhadi.[YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)
* Ross Girshick (Microsoft Research)[Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
* Long, Jonathan, Evan Shelhamer, and Trevor Darrell.[Fully convolutional networks for semantic segmentation](https://arxiv.org/pdf/1605.06211.pdf)
* Jégou, Simon, et al.[The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf)

