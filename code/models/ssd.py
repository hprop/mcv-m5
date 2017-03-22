from keras.layers import (Input, Flatten, Convolution2D, MaxPooling2D,
                          AtrousConvolution2D, GlobalAveragePooling2D)

from layers.ssd_layers import Normalize, PriorBox


# Sources:
# - TODO: add paper reference
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
    net['pool1'] = MaxPooling2D((2, 2),
                                strides=(2, 2),
                                border_mode='same',
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
    net['pool2'] = MaxPooling2D((2, 2),
                                strides=(2, 2),
                                border_mode='same',
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
    net['pool3'] = MaxPooling2D((2, 2),
                                strides=(2, 2),
                                border_mode='same',
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
    net['pool4'] = MaxPooling2D((2, 2),
                                strides=(2, 2),
                                border_mode='same',
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
    net['pool5'] = MaxPooling2D((3, 3),
                                strides=(1, 1),
                                border_mode='same',
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

    # Add extra layer on top of conv4_3 to normalize its output according to
    # the paper
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])

    return net


def create_prior(layer_name, n_priors, min_size, max_size, aspect_ratios,
                 variances):
    return dict(layer_name=layer_name, n_priors=n_priors, min_size=min_size,
                max_size=max_size, aspect_ratios=aspect_ratios,
                variances=variances)


def build_ssd300(input_shape, n_classes):

    vgg16 = vgg16_base_network(input_shape)

    variances = [.1, .1, .2, .2]
    priors = [create_prior('conv4_3_norm', 3, 30., None, [2], variances),
              create_prior('conv7', 6, 60., 114., [2, 3], variances),
              create_prior('conv8_2', 6, 114., 168., [2, 3], variances),
              create_prior('conv9_2', 6, 168., 222., [2, 3], variances),
              create_prior('conv10_2', 6, 168., 222., [2, 3], variances),
              create_prior('conv11_2', 6, 276., 330., [2, 3], variances)]

    ssd300 = build_ssd(input_shape, n_classes, vgg16, priors)

    return ssd300




def build_ssd(input_shape, n_classes, base_network, priors):
    """
    input_shape: (h, w)


    priors: dict with keys being the names of layers used as feature extractors,
    and values being the priors (bboxes info) corresponding to each layer.

    """

    net = dict(base_network)


    ## For each layer taking as a prediction input:

    base_layer = 'conv4_3_norm'
    num_priors = 3
    prior_min_size = 30.
    prior_max_size = None
    prior_aspect_ratios = [2]
    prior_variances = [0.1, 0.1, 0.2, 0.2]

    # Bounding box locations
    loc_layer = base_layer + '_loc'
    loc_layer_flat = base_layer + '_loc_flat'

    net[loc_layer] = Convolution2D(num_priors * 4, 3, 3,
                                   border_mode='same',
                                   name=loc_layer)(net[base_layer])
    net[loc_layer_flat] = Flatten(name=loc_layer_flat)(net[loc_layer])

    # Class confidences for each bounding box
    conf_layer = base_layer + '_conf'
    conf_layer_flat = base_layer + '_conf_flat'

    net[conf_layer] = Convolution2D(num_priors * n_classes, 3, 3,
                                    border_mode='same',
                                    name=conf_layer)(net[base_layer])
    net[conf_layer_flat] = Flatten(name=conf_layer_flat)(net[conf_layer])

    # Bounding box priors (one for each layer)
    priors_layer = base_layer + '_priors'

    net[priors_layer] = PriorBox([input_shape[1], input_shape[0]],
                                 prior_min_size,
                                 max_size=prior_max_size,
                                 aspect_ratios=prior_aspect_ratios,
                                 variances=prior_variances,
                                 name=priors_layer)(net[base_layer])
