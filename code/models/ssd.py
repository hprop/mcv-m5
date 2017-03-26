from keras.models import Model
from keras.layers import (Input, Flatten, Reshape, merge, Activation,
                          Convolution2D, MaxPooling2D, AtrousConvolution2D,
                          GlobalAveragePooling2D)

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
    # net['pool11'] = GlobalAveragePooling2D(name='pool11')(net['conv11_2'])

    # Add extra layer on top of conv4_3 to normalize its output according to
    # the paper
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])

    return net


def create_priors(layer_name, n_boxes, min_size, max_size, aspect_ratios,
                 variances):
    return dict(layer_name=layer_name, n_boxes=n_boxes, min_size=min_size,
                max_size=max_size, aspect_ratios=aspect_ratios,
                variances=variances)


def build_ssd300(input_shape, n_classes):

    vgg16 = vgg16_base_network(input_shape)

    variances = [.1, .1, .2, .2]
    priors = [create_priors('conv4_3_norm', 3, 10., None, [2], variances),
              create_priors('conv7', 5, 60., 114., [2, 3], variances),
              create_priors('conv8_2', 5, 114., 168., [2, 3], variances),
              create_priors('conv9_2', 5, 168., 222., [2, 3], variances),
              create_priors('conv10_2', 5, 222., 276., [2, 3], variances),
              create_priors('conv11_2', 5, 276., 330., [2, 3], variances)]

    ssd300 = build_ssd(input_shape, n_classes, vgg16, priors)

    return ssd300


def build_ssd(input_shape, n_classes, base_network, priors):
    """input_shape: (h, w, c)


    priors: dict with keys being the names of layers used as feature
    extractors, and values being the priors (bboxes info) corresponding to each
    layer.

    """

    net = dict(base_network)

    # Build the prediction layers on top of the base network
    for p in priors:
        base_layer = p['layer_name']
        num_priors = p['n_boxes']
        prior_min_size = p['min_size']
        prior_max_size = p['max_size']
        prior_aspect_ratios = p['aspect_ratios']
        prior_variances = p['variances']

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

    # Build the top layer
    feature_layers = [p['layer_name'] for p in priors]
    loc_layers = [net[name + '_loc_flat'] for name in feature_layers]
    conf_layers = [net[name + '_conf_flat'] for name in feature_layers]
    priors_layers = [net[name + '_priors'] for name in feature_layers]

    # TODO: this is theano-compatible?
    net['pred_loc'] = merge(loc_layers, mode='concat', concat_axis=1,
                            name='pred_loc')

    n_boxes = net['pred_loc']._keras_shape[-1] // 4
    net['pred_loc'] = Reshape((n_boxes, 4),
                              name='pred_loc_resh')(net['pred_loc'])

    net['pred_conf'] = merge(conf_layers, mode='concat', concat_axis=1,
                             name='pred_conf')
    net['pred_conf'] = Reshape((n_boxes, n_classes),
                               name='pred_conf_resh')(net['pred_conf'])
    net['pred_conf'] = Activation('softmax',
                                  name='pred_conf_final')(net['pred_conf'])

    net['pred_prior'] = merge(priors_layers, mode='concat', concat_axis=1,
                              name='pred_prior')

    net['pred_all'] = merge([net['pred_loc'],
                             net['pred_conf'],
                             net['pred_prior']],
                            mode='concat',
                            concat_axis=2,
                            name='pred_all')

    model = Model(net['input'], net['pred_all'])

    return model
