from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
import math


# Sources:
#
# * Original paper: https://arxiv.org/pdf/1608.06993.pdf
# * https://github.com/liuzhuang13/DenseNet
# * https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet
# * https://github.com/robertomest/convnet-study
# * https://github.com/titu1994/DenseNet



def transition(x, n_filter, dropout=None, weight_decay=1E-4):
    '''
    Transition layer on the Densenet

    :param x: Input features (4D tensor -> numpy array)
    :param n_filter: number of feature maps on the 2DConv
    :param dropout: None or float. When it is not None, apply dropout layers
        after each 2DConv layer with probability to drop equal to this value.
    :param weight_decay: weight decay on the regularizers for BN and 2DConv

    :return: Output features (4D tensor -> numpy array)

    '''
    x = BatchNormalization(mode=0,
                           axis=1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Convolution2D(n_filter, 1, 1,
                      init='he_uniform',
                      border_mode='same',
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)

    if dropout is not None:
        x = Dropout(dropout)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def denseblock(x, n_layers, n_filter, n_bottleneck=None, dropout=None,
               weight_decay=1E-4):
    '''
    Dense block on the Densenet

    :param x: Input features (4D tensor -> numpy array)
    :param n_layers: Number of layers on the block
    :param n_filter: number of 3x3 feature maps on the 2DConv layers
    :param n_bottleneck: number of 1x1 feature maps on the 2DConv bottleneck
        layers. If None, no bottleneck algorithm will be applied.
    :param dropout: None or float. When it is not None, apply dropout layers
        after each 2DConv layer with probability to drop equal to this value.
    :param weight_decay: weight decay on the regularizers for BN and 2DConv

    :return: Output features (4D tensor -> numpy array)

    '''
    list_feat = [x]

    if K.image_dim_ordering() == 'th':
        concat_axis = 1
    elif K.image_dim_ordering() == 'tf':
        concat_axis = -1

    for i in range(n_layers):
        if n_bottleneck is not None:
            x = BatchNormalization(mode=0,
                                   axis=1,
                                   gamma_regularizer=l2(weight_decay),
                                   beta_regularizer=l2(weight_decay))(x)
            x = Activation('relu')(x)
            x = Convolution2D(n_bottleneck, 1, 1,
                              init='he_uniform',
                              border_mode='same',
                              bias=False,
                              W_regularizer=l2(weight_decay))(x)
            if dropout is not None:
                x = Dropout(dropout)(x)

        x = BatchNormalization(mode=0,
                               axis=1,
                               gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Convolution2D(n_filter, 3, 3,
                          init='he_uniform',
                          border_mode='same',
                          bias=False,
                          W_regularizer=l2(weight_decay))(x)
        if dropout is not None:
            x = Dropout(dropout)(x)

        list_feat.append(x)
        x = merge(list_feat, mode='concat', concat_axis=concat_axis)

    return x


def classification_block(x, n_classes, weight_decay=1E-4):
    '''
    Classification block on the dataset

    :param x: Input features (4D tensor -> numpy array)
    :param n_classes: Number of classes on the dataset, it is used by softmax
    :param weight_decay: weight decay on the regularizers for BN and 2DConv

    :return: Output features (4D tensor -> numpy array)

    '''
    x = BatchNormalization(mode=0,
                           axis=1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D(dim_ordering=K.image_dim_ordering)(x)
    x = Dense(n_classes,
              activation='softmax',
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)
    return x


def build_densenet(img_shape=(3, 224, 224), n_classes=1000,
                   layers_in_dense_block=[6, 12, 24, 16], initial_filters=16,
                   growth_rate=32, n_bottleneck=None, compression=1.,
                   dropout=None, weight_decay=0.):
    '''
    Returns a Densenet model

    :param img_shape: Shape of the input images for the network. Tuple of 3
        containing channels, roes and columns.
    :param n_classes: Number of classes on the dataset.
    :param layers_in_dense_block: Number of layers on each dense block. List.
    :param initial_filters: Number of filters applied on the first 2DConv layer.
    :param growth_rate: Growth rate of the network.
    :param n_bottleneck: number of 1x1 feature maps on the 2DConv bottleneck
        layers. If None, no bottleneck algorithm will be applied.
    :param compression: Compression rate for the transition layers.
    :param dropout: None or float. When it is not None, apply dropout layers
        after each 2DConv layer with probability to drop equal to this value.
    :param weight_decay: weight decay on the regularizers for BN and 2DConv.

    :return: the Densenet model

    '''

    model_input = Input(shape=img_shape)

    n_filters = initial_filters

    x = Convolution2D(n_filters, 7, 7,
                      init='he_uniform',
                      border_mode='same',
                      name='first_layer',
                      bias=False)(model_input)

    for block in layers_in_dense_block[:-1]:
        x = denseblock(x, block, growth_rate, n_bottleneck=n_bottleneck,
                       dropout=dropout, weight_decay=weight_decay)
        n_filters += math.floor(compression*growth_rate*block)
        x = transition(x, n_filters, dropout=dropout,
                       weight_decay=weight_decay)

    x = denseblock(x, layers_in_dense_block[-1], growth_rate,
                   n_bottleneck=n_bottleneck, dropout=dropout,
                   weight_decay=weight_decay)

    x = classification_block(x, n_classes, weight_decay=weight_decay)

    model = Model(input=[model_input], output=[x])

    return model
