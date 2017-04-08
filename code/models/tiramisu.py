from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D,ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from layers.deconv import Deconvolution2D
from layers.ourlayers import (CropLayer2D, NdSoftmax)
from keras import backend as K
dim_ordering = K.image_dim_ordering()
import math
import tensorflow as tf

#skip connexions employed in upsampling
skip_connection_list = []
block_to_upsample = []

def transition_down(x, n_filter, dropout=None, weight_decay=1E-4):
    '''
    Transition layer on the Densenet

    :param x: Input features (4D tensor -> numpy array)
    :param n_filter: number of feature maps on the 2DConv
    :param dropout: None or float. When it is not None, apply dropout layers
        after each 2DConv layer with probability to drop equal to this value.
    :param weight_decay: weight decay on the regularizers for BN and 2DConv

    :return: Output features (4D tensor -> numpy array)

    '''
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Convolution2D(n_filter, 1, 1,
                      init='he_uniform',
                      border_mode='same',
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)

    if dropout is not None:
        x = Dropout(dropout)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    return x

def transition_up(skip_conn_lev, x, n_filter, weight_decay=1E-4):
    '''
    Transition Up layer on the Densenet

    :param skip_conn_lev: Skip connections (4D tensor -> numpy array)
    :param x: Input features (4D tensor -> numpy array)
    :param n_filter: number of feature maps on the 2DConv
    :param weight_decay: weight decay on the regularizers for BN and 2DConv

    :return: Output features (4D tensor -> numpy array)

    '''
    print("skip_conn"+str(skip_conn_lev))
    print("block_to_upsample"+str(x))
    print("filters"+str(n_filter))
    if K.image_dim_ordering() == 'th':
        concat_axis = 1
    elif K.image_dim_ordering() == 'tf':
        concat_axis = -1

    x = merge(x, mode='concat', concat_axis=concat_axis)

    #x = Deconvolution2D(n_filter, 3, 3,x._keras_shape,
    #                  init='he_uniform',
    #                  border_mode='same',
    #                  subsample=(2, 2),
    #                  bias=False,name='deconvolution',
    #                  W_regularizer=l2(weight_decay))(x)
    x = Deconvolution2D(n_filter, 3, 3,x._keras_shape,subsample=(2, 2))(x)
    tf.concat([x,skip_conn_lev],3)

    return x


def denseblock(x, n_layers, n_filter, n_bottleneck=None, dropout=None,
               weight_decay=1E-4,upsample=False):
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
    global block_to_upsample
    global skip_connection_list
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
 
        #x = BatchNormalization(mode=0,
        #                       axis=1,
        #                       gamma_regularizer=l2(weight_decay),
        #                       beta_regularizer=l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(n_filter, 3, 3,
                          init='he_uniform',
                          border_mode='same',
                          bias=False,
                          W_regularizer=l2(weight_decay))(x)
        if dropout is not None:
            x = Dropout(dropout)(x)

        list_feat.append(x)
        
        if upsample == True:
            block_to_upsample.append(x)
        else:
            skip_connection_list.append(x)
        x = merge(list_feat, mode='concat', concat_axis=concat_axis)

    return x


def segmentation_block(x,n_classes, weight_decay=0.):
    '''
    Classification block on the dataset

    :param x: Input features (4D tensor -> numpy array)

    :return: Output features (4D tensor -> numpy array)

    '''
    x = Convolution2D(n_classes, 1, 1,
                      init='he_uniform',
                      border_mode='same',
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    x = NdSoftmax()(x)
    return x

def build_tiramisu(img_shape=(3, None, None), n_classes=11,
                   layers_in_dense_block=[4, 5, 7, 10, 12, 15], initial_filters=48,
                   growth_rate=16, n_bottleneck=None, compression=1.,
                   dropout=0.2, weight_decay=0.):
    '''
    Returns a Tiramisu model

    :param img_shape: Shape of the input images for the network. Tuple of 3
        containing channels, rows and columns.
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
    global block_to_upsample
    global skip_connection_list
    model_input = Input(shape=img_shape)
    padded = ZeroPadding2D(padding=(100, 100), name='pad100')(model_input)
    n_filters = initial_filters

    x = Convolution2D(n_filters, 3, 3,
                      init='he_uniform',
                      border_mode='same',
                      name='first_layer',
                      bias=False)(padded)

    # Downsampling path #

    for block in layers_in_dense_block[:-1]:
        x = denseblock(x, block, growth_rate, n_bottleneck=n_bottleneck,
                       dropout=dropout, weight_decay=weight_decay)
        n_filters += math.floor(compression*growth_rate*block)
        x = transition_down(x, n_filters, dropout=dropout,
                       weight_decay=weight_decay)

    #reverse array. when iterating first upsampling concatenation picks skipped connections from last layer
    #skip_connection_list = skip_connection_list[::-1]
    skip_connection_list = skip_connection_list[::-1]
    # bottleneck #

    # We store now the output of the next dense block in a list. We will only upsample these new feature maps

    x = denseblock(x, layers_in_dense_block[-1], growth_rate,
                   n_bottleneck=n_bottleneck, dropout=dropout,
                   weight_decay=weight_decay,upsample=True)
    # Upsampling path
    #for block in reversed(layers_in_dense_block[:-1]):
    #skip_connections layers [12, 10, 7, 5, 4] since last block (15 layers) is already
    # added to block_to_upsample
    print("upsampling")
    for idx, block in enumerate(reversed(layers_in_dense_block)):
        if idx != 0:
            curr_filters = block * growth_rate
            n_filters_keep = int(math.floor(curr_filters + n_filters + prev_n_filters))
            print("Block:" + str(idx) + " - Layers: " + str(block) +
                  " - Number filters:" + str(n_filters_keep))
            x = transition_up(skiped_maps, block_to_upsample, n_filters_keep)
            block_to_upsample = []
            x = denseblock(x, block, growth_rate, n_bottleneck=n_bottleneck,
                           dropout=dropout, weight_decay=weight_decay, upsample=True)
        skiped_maps = skip_connection_list[idx]
        prev_n_filters = math.floor(compression * growth_rate * block)
        n_filters -= math.floor(compression * growth_rate * block)



    # Segmentation block #
    x = segmentation_block(x,n_classes, weight_decay=weight_decay)

    model = Model(input=[model_input], output=[x])

    return model

if __name__ == '__main__':
    input_shape = [3, 224, 224]
    print (' > Building')
    model = build_tiramisu(input_shape, 11)
    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()

