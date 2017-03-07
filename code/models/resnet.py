from keras.models import Model
from keras.applications import resnet50
from keras.layers import Flatten, Dense


def build_resnet50(img_shape=(3, 224, 224), n_classes=1000, l2_reg=0.,
                   load_pretrained=False, freeze_layers_from='base_model'):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None

    # Get base model
    base_model = resnet50.ResNet50(include_top=False,
                                   weights=weights,
                                   input_shape=img_shape)

    # Add final layers
    x = base_model.output
    x = Flatten(name='flatten')(x)
    predictions = Dense(n_classes, activation='softmax', name='predictions')(x)

    # Declare final model
    model = Model(input=base_model.input, output=predictions)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True

    return model
