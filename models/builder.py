import tensorflow as tf
from chambers.models import *

def get_model(model_name, input_shape, **kwargs):
    input_layer = tf.keras.layers.Input(input_shape)

    # Classification
    if model_name == "ResNet50":
        return ResNet50(inputs=input_layer, **kwargs)
    elif model_name == "ResNet101":
        return ResNet101(inputs=input_layer, **kwargs)
    elif model_name == "ResNet152":
        return ResNet152(inputs=input_layer, **kwargs)

    # Segmentation
    elif model_name == "UNet":
        return UNet(inputs=input_layer, **kwargs)
    elif model_name == "FRRN-A":
        return FRRN_A(inputs=input_layer, **kwargs)
    elif model_name == "FRRN-B":
        return FRRN_B(inputs=input_layer, **kwargs)
    elif model_name == "ResNet50_FPN":
        return ResNet50_FPN(inputs=input_layer, **kwargs)
    elif model_name == "ResNet101_FPN":
        return ResNet101_FPN(inputs=input_layer, **kwargs)
    elif model_name == "ResNet152_FPN":
        return ResNet152_FPN(inputs=input_layer, **kwargs)

    # GAN
    elif model_name == "Pix2Pix":
        return Pix2Pix(inputs=input_layer)

    else:
        raise ValueError("Invalid model")