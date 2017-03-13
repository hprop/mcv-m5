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


def transition(x, n_filter, weight_decay=1E-4):

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
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def denseblock(x, n_layers, n_filter, n_bottleneck=None, weight_decay=1E-4):

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

        list_feat.append(x)
        x = merge(list_feat, mode='concat', concat_axis=concat_axis)

    return x


def classification_block(x, n_classes, weight_decay=1E-4):

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
                   layers_in_dense_block=[6, 12, 24, 16], n_filters=16,
                   growth_rate=32, n_bottleneck=None, weight_decay=0.):

    model_input = Input(shape=img_shape)

    n_dense_block = len(layers_in_dense_block)

    x = Convolution2D(n_filters, 7, 7,
                      init='he_uniform',
                      border_mode='same',
                      name='first_layer',
                      bias=False)(model_input)

    for i in range(n_dense_block - 1):
        x = denseblock(x, layers_in_dense_block[i], growth_rate,
                       weight_decay=weight_decay)
        n_filters += (growth_rate*layers_in_dense_block[i])
        x = transition(x, n_filters, weight_decay=weight_decay)

    x = denseblock(x, layers_in_dense_block[n_dense_block-1], n_filters,
                   n_bottleneck=n_bottleneck, weight_decay=weight_decay)

    x = classification_block(x, n_classes, weight_decay=weight_decay)

    model = Model(input=[model_input], output=[x])

    return model