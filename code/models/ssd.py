from keras.layers import (Input, Convolution2D, MaxPooling2D,
                          AtrousConvolution2D, GlobalAveragePooling2D)


# Sources:
# - https://github.com/rykov8/ssd_keras


def vgg16_base_network(input_shape=None):
    """VGG16 base model used in SSD paper"""

    net = {}

    # Block 1
    net['input'] = Input(shape=input_shape)
    net['conv1_1'] = Convolution2D(64, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv1_1')(net['input'])
    net['conv1_2'] = Convolution2D(64, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv1_2')(net['conv1_1'])
    net['pool1'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool1')(net['conv1_2'])

    # Block 2
    net['conv2_1'] = Convolution2D(128, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv2_1')(net['pool1'])
    net['conv2_2'] = Convolution2D(128, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv2_2')(net['conv2_1'])
    net['pool2'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool2')(net['conv2_2'])

    # Block 3
    net['conv3_1'] = Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_1')(net['pool2'])
    net['conv3_2'] = Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_3')(net['conv3_2'])
    net['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool3')(net['conv3_3'])

    # Block 4
    net['conv4_1'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_1')(net['pool3'])
    net['conv4_2'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_3')(net['conv4_2'])
    net['pool4'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool4')(net['conv4_3'])

    # Block 5
    net['conv5_1'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_1')(net['pool4'])
    net['conv5_2'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_3')(net['conv5_2'])
    net['pool5'] = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same',
                                name='pool5')(net['conv5_3'])

    # Block 6
    net['conv6'] = AtrousConvolution2D(1024, 3, 3,
                                       atrous_rate=(6, 6),
                                       activation='relu',
                                       border_mode='same',
                                       name='conv6')(net['pool5'])

    # Block 7
    net['conv7'] = Convolution2D(1024, 1, 1,
                                 activation='relu',
                                 border_mode='same',
                                 name='conv7')(net['conv6'])

    # Block 8
    net['conv8_1'] = Convolution2D(256, 1, 1,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv8_1')(net['conv7'])
    net['conv8_2'] = Convolution2D(512, 3, 3,
                                   subsample=(2, 2),
                                   activation='relu',
                                   border_mode='same',
                                   name='conv8_2')(net['conv8_1'])

    # Block 9
    net['conv9_1'] = Convolution2D(128, 1, 1,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv9_1')(net['conv8_2'])
    net['conv9_2'] = Convolution2D(256, 3, 3,
                                   subsample=(2, 2),
                                   activation='relu',
                                   border_mode='same',
                                   name='conv9_2')(net['conv9_1'])

    # Block 10
    net['conv10_1'] = Convolution2D(128, 1, 1,
                                    activation='relu',
                                    border_mode='same',
                                    name='conv10_1')(net['conv9_2'])
    net['conv10_2'] = Convolution2D(256, 3, 3,
                                    activation='relu',
                                    border_mode='same',
                                    name='conv10_2')(net['conv10_1'])

    # Block 11
    net['conv11_1'] = Convolution2D(128, 1, 1,
                                    activation='relu',
                                    border_mode='same',
                                    name='conv11_1')(net['conv10_2'])
    net['conv11_2'] = Convolution2D(256, 3, 3,
                                    activation='relu',
                                    border_mode='same',
                                    name='conv11_2')(net['conv11_1'])

    # TODO: remove final GlobalAveragePooling2D?
    ## Global Average Pooling is not used in the original paper.
    ## Added here to obtain a final feature maps resolution of 1x1 when
    ## image size is > 300x300.
    net['pool11'] = GlobalAveragePooling2D(name='pool11')(net['conv11_2'])

    return net


def build_ssd(base_network, priors):
    """priors: dict with keys being the names of layers used as feature extractors,
    and values being the priors (bboxes info) corresponding to each layer.

    """
    pass
