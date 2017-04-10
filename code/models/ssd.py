from keras.models import Model
from keras.layers import (Input, Flatten, Reshape, merge, Activation,
                          Convolution2D, MaxPooling2D, AtrousConvolution2D,
                          GlobalAveragePooling2D)

from layers.ssd_layers import Normalize, PriorBox
from models import resnet
from models import vgg


# Sources:
# - Paper: https://arxiv.org/pdf/1512.02325.pdf
# - Code based on: https://github.com/rykov8/ssd_keras


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


def vgg16_pretrained_base_network(input_shape):
    """VGG16 base model to use with pretrained weights

    In order to allow loading the weights, the vgg16 network is built with the
    layer names found in the vgg16 model provided by keras.

    `input_shape` must be a tuple (height, width, channels).

    Returns the layers in a dict which must be pass to the build_ssd()
    function.

    """
    net = {}

    # Block 1
    net['input'] = Input(shape=input_shape)
    net['block1_conv1'] = Convolution2D(64, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='block1_conv1')(net['input'])
    net['block1_conv2'] = Convolution2D(64, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='block1_conv2')(net['block1_conv1'])
    net['block1_pool'] = MaxPooling2D((2, 2),
                                      strides=(2, 2),
                                      border_mode='same',
                                      name='block1_pool')(net['block1_conv2'])

    # Block 2
    net['block2_conv1'] = Convolution2D(128, 3, 3,
                                        activation='relu',
                                        border_mode='same',
                                        name='block2_conv1')(net['block1_pool'])
    net['block2_conv2'] = Convolution2D(128, 3, 3,
                                        activation='relu',
                                        border_mode='same',
                                        name='block2_conv2')(net['block2_conv1'])
    net['block2_pool'] = MaxPooling2D((2, 2),
                                      strides=(2, 2),
                                      border_mode='same',
                                      name='block2_pool')(net['block2_conv2'])

    # Block 3
    net['block3_conv1'] = Convolution2D(256, 3, 3,
                                        activation='relu',
                                        border_mode='same',
                                        name='block3_conv1')(net['block2_pool'])
    net['block3_conv2'] = Convolution2D(256, 3, 3,
                                        activation='relu',
                                        border_mode='same',
                                        name='block3_conv2')(net['block3_conv1'])
    net['block3_conv3'] = Convolution2D(256, 3, 3,
                                        activation='relu',
                                        border_mode='same',
                                        name='block3_conv3')(net['block3_conv2'])
    net['block3_pool'] = MaxPooling2D((2, 2),
                                      strides=(2, 2),
                                      border_mode='same',
                                      name='block3_pool')(net['block3_conv3'])

    # Block 4
    net['block4_conv1'] = Convolution2D(512, 3, 3,
                                        activation='relu',
                                        border_mode='same',
                                        name='block4_conv1')(net['block3_pool'])
    net['block4_conv2'] = Convolution2D(512, 3, 3,
                                        activation='relu',
                                        border_mode='same',
                                        name='block4_conv2')(net['block4_conv1'])
    net['block4_conv3'] = Convolution2D(512, 3, 3,
                                        activation='relu',
                                        border_mode='same',
                                        name='block4_conv3')(net['block4_conv2'])
    net['block4_pool'] = MaxPooling2D((2, 2),
                                strides=(2, 2),
                                border_mode='same',
                                name='block4_pool')(net['block4_conv3'])

    # Block 5
    net['block5_conv1'] = Convolution2D(512, 3, 3,
                                        activation='relu',
                                        border_mode='same',
                                        name='block5_conv1')(net['block4_pool'])
    net['block5_conv2'] = Convolution2D(512, 3, 3,
                                        activation='relu',
                                        border_mode='same',
                                        name='block5_conv2')(net['block5_conv1'])
    net['block5_conv3'] = Convolution2D(512, 3, 3,
                                        activation='relu',
                                        border_mode='same',
                                        name='block5_conv3')(net['block5_conv2'])
    net['block5_pool'] = MaxPooling2D((3, 3),
                                      strides=(1, 1),
                                      border_mode='same',
                                      name='block5_pool')(net['block5_conv3'])

    # Block 6
    net['conv6'] = AtrousConvolution2D(1024, 3, 3,
                                       atrous_rate=(6, 6),
                                       activation='relu',
                                       border_mode='same',
                                       name='conv6')(net['block5_pool'])

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

    # Add extra layer on top of conv4_3 to normalize its output according to
    # the paper
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['block4_conv3'])

    return net


def vgg16_tt100k_base_network(input_shape, pretrained_weigths):
    """VGG16 base model with pretrained weights

    `pretrained_weigths` is an hdf5 file with the weights from a VGG16 model
    pretrained on the TT100K dataset.

    NOTE: expected input_shape for the model must be (64, 64, 3). It is the
    input shape we used for our trained vgg16 models.

    `input_shape` is a tuple (height, width, channels).

    Return a dict with the layers to use from the base model.

    """
    base_model = vgg.build_vgg(img_shape=(64, 64, 3), n_classes=45,
                               n_layers=16, freeze_layers_from=None)

    base_model.load_weights(pretrained_weigths)

    net = {}

    net['input'] = base_model.input
    net['conv4_3'] = base_model.get_layer('block4_conv3').output
    net['pool5'] = base_model.get_layer('block5_pool').output

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

    # Add extra layer on top of conv4_3 to normalize its output according to
    # the paper
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])

    return net


def create_priors(layer_name, min_size, max_size, aspect_ratios,
                 variances):
    """Create the priors to be applied in a layer

    Parameters
    ----------
    layer_name: str
        Name of the layer in the base network where the priors are applied.

    min_size: float
        Minimum size in pixels the priors can have.

    max_size: float
        Maximum size in pixels the priors can have.

    aspect_ratios: list of float
        Aspect ratios of the priors to be used.

    variances:
        TODO: review

    The priors are passed to the build_ssd() function, and the final number of
    priors will be: 2 * len(aspect_ratios) + 1. Two priors per each aspect
    ratio specified in `aspect_ratios' (one for the actual value, other for its
    inverse 1/aspect), plus a prior with aspect ratio 1, which is internally
    added in the build_ssd() function.

    """
    n_boxes = len(aspect_ratios) * 2 + 1
    return dict(layer_name=layer_name, n_boxes=n_boxes, min_size=min_size,
                max_size=max_size, aspect_ratios=aspect_ratios,
                variances=variances)


def build_ssd(input_shape, n_classes, base_network, priors):
    """input_shape: (h, w, c)


    priors: list of priors created through create_priors() function.

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


def build_ssd300(input_shape, n_classes):
    """Create a SSD300 model

    `input_shape' is in the form (w, h, c), and `n_classes' includes the
    background class (so "n_classes = positive_classes + 1").

    """
    vgg16 = vgg16_base_network(input_shape)

    variances = [.1, .1, .2, .2]
    priors = [create_priors('conv4_3_norm', 10., None, [1.25, 1.5], variances),
              create_priors('conv7', 60., 114., [1.5], variances),
              create_priors('conv8_2', 114., 168., [1.5], variances),
              create_priors('conv9_2', 168., 222., [1.5], variances),
              create_priors('conv10_2', 222., 276., [1.5], variances),
              create_priors('conv11_2', 276., 330., [1.5], variances)]

    ssd300 = build_ssd(input_shape, n_classes, vgg16, priors)

    return ssd300


def build_ssd300_pretrained(input_shape, n_classes):
    """Create a SSD300 model to use with pretrained weights from a vgg16 keras model

    In order to allow loading the weights, the vgg16 base network is built with the
    layer names found in the vgg16 model provided by keras.

    `input_shape` is a tuple in the form (height, width, channels).
    `n_classes` must be include the background class.

    """
    vgg16 = vgg16_pretrained_base_network(input_shape)

    variances = [.1, .1, .2, .2]
    priors = [create_priors('conv4_3_norm', 10., None, [1.25, 1.5], variances),
              create_priors('conv7', 60., 114., [1.5], variances),
              create_priors('conv8_2', 114., 168., [1.5], variances),
              create_priors('conv9_2', 168., 222., [1.5], variances),
              create_priors('conv10_2', 222., 276., [1.5], variances),
              create_priors('conv11_2', 276., 330., [1.5], variances)]

    sdd = build_ssd(input_shape, n_classes, vgg16, priors)

    return sdd


def build_ssd_resnet50(input_shape, n_classes, load_pretrained=None):
    """Create a SSD model with resnet50 as the base network

    When using with pretrained weights, they must come from a keras ResNet50
    model or from models.resnet.build_resnet50().

    Parameters
    ----------
    input_shape : tuple
        Input shape for the model in the form (w, h, c).

    n_classes : int
        Number of classes to detect by the model.

    load_pretrained : str or None
        If None, no pretrained weights are used. Otherwise, it is assumed the
        value is a path to a hdf5 file with the weights.

    """
    base_model = resnet.build_resnet50(img_shape=input_shape,
                                       n_classes=n_classes,
                                       freeze_layers_from=None,
                                       include_top=False)

    if load_pretrained:
        base_model.load_weights(load_pretrained, by_name=True)

    resnet50 = {}
    resnet50['input'] = base_model.input
    resnet50['block2'] = base_model.get_layer('res3a_branch2a').input
    resnet50['block3'] = base_model.get_layer('res4a_branch2a').input
    resnet50['block4'] = base_model.get_layer('res5a_branch2a').input
    resnet50['block5'] = base_model.get_layer('avg_pool').input

    # Take as base layers the resnet merge layers from blocks 2, 3, 4 and 5
    variances = [.1, .1, .2, .2]
    priors = [create_priors('block2', 10., None, [1.25, 1.5], variances),
              create_priors('block3', 82., 157., [1.5], variances),
              create_priors('block4', 157., 235., [1.5], variances),
              create_priors('block5', 235., 320., [1.5], variances)]

    ssd = build_ssd(input_shape, n_classes, resnet50, priors)

    return ssd
