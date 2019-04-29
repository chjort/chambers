import numpy as np

def pad_to_shape(layer1, layer2):
    diff_h = layer2.shape.as_list()[1] - layer1.shape.as_list()[1]
    diff_w = layer2.shape.as_list()[2] - layer1.shape.as_list()[2]
    shape = ((0, diff_h), (0, diff_w))

    return shape


def pad_to_factor(inputs, factor):
    input_shape = inputs.shape.as_list()
    height, width = input_shape[1], input_shape[2]

    if height % factor != 0:
        pad_height_to = (height // factor + 1) * factor
        diff_h = pad_height_to - height
        pad_h = (int(np.floor(diff_h/2)), int(np.ceil(diff_h/2)))
    else:
        pad_h = (0, 0)

    if width % factor != 0:
        pad_width_to = (width // factor + 1) * factor
        diff_w = pad_width_to - width
        pad_w = (int(np.floor(diff_w/2)), int(np.ceil(diff_w/2)))
    else:
        pad_w = (0, 0)

    return pad_h, pad_w