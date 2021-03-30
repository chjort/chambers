import collections

from tensorflow.python.keras import backend, layers, models
from tensorflow.python.keras import utils as keras_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.lib.io import file_io

from chambers.augmentations import ImageNetNormalization

ModelParams = collections.namedtuple(
    'ModelParams',
    ['model_name', 'repetitions', 'residual_block', 'groups',
     'reduction', 'init_filters', 'input_3x3', 'dropout']
)

BASE_WEIGHTS_PATH = (
    'https://github.com/chjort/chambers/releases/download/v1.0/')
WEIGHTS_HASHES = {
    'seresnet50':
        ('ff0ce1ed5accaad05d113ecef2d29149', '043777781b0d5ca756474d60bf115ef1'),
    'seresnet101':
        ('5c31adee48c82a66a32dee3d442f5be8', '1c373b0c196918713da86951d1239007'),
    'seresnet152':
        ('96fc14e3a939d4627b0174a0e80c7371', 'f58d4c1a511c7445ab9a2c2b83ee4e7b'),
    'seresnext50':
        ('5310dcd58ed573aecdab99f8df1121d5', 'b0f23d2e1cd406d67335fb92d85cc279'),
    'seresnext101':
        ('be5b26b697a0f7f11efaa1bb6272fc84', 'e48708cbe40071cc3356016c37f6c9c7'),
    'senet154':
        ('c8eac0e1940ea4d8a2e0b2eb0cdf4e75', 'd854ff2cd7e6a87b05a8124cd283e0f2'),
}


# -------------------------------------------------------------------------
#   Helpers functions
# -------------------------------------------------------------------------

def get_bn_params(**params):
    axis = 3 if backend.image_data_format() == 'channels_last' else 1
    default_bn_params = {
        'axis': axis,
        'epsilon': 9.999999747378752e-06,
    }
    default_bn_params.update(params)
    return default_bn_params


def get_num_channels(tensor):
    channels_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    return backend.int_shape(tensor)[channels_axis]


# -------------------------------------------------------------------------
#   Common blocks
# -------------------------------------------------------------------------
def slice_tensor(x, start, stop, axis):
    if axis == 3:
        return x[:, :, :, start:stop]
    elif axis == 1:
        return x[:, start:stop, :, :]
    else:
        raise ValueError("Slice axis should be in (1, 3), got {}.".format(axis))


def GroupConv2D(filters,
                kernel_size,
                strides=(1, 1),
                groups=32,
                kernel_initializer='he_uniform',
                use_bias=True,
                activation='linear',
                padding='valid',
                **kwargs):
    """
    Grouped Convolution Layer implemented as a Slice,
    Conv2D and Concatenate layers. Split filters to groups, apply Conv2D and concatenate back.

    Args:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer, specifying the stride
            length of the convolution.
        groups: Integer, number of groups to split input filters to.
        kernel_initializer: Regularizer function applied to the kernel weights matrix.
        use_bias: Boolean, whether the layer uses a bias vector.
        activation: Activation function to use (see activations).
            If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        padding: one of "valid" or "same" (case-insensitive).

    Input shape:
        4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".

    Output shape:
        4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format is "channels_last".
        rows and cols values might have changed due to padding.

    """

    slice_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor):
        inp_ch = int(backend.int_shape(input_tensor)[-1] // groups)  # input grouped channels
        out_ch = int(filters // groups)  # output grouped channels

        blocks = []
        for c in range(groups):
            slice_arguments = {
                'start': c * inp_ch,
                'stop': (c + 1) * inp_ch,
                'axis': slice_axis,
            }
            x = layers.Lambda(slice_tensor, arguments=slice_arguments)(input_tensor)
            x = layers.Conv2D(out_ch,
                              kernel_size,
                              strides=strides,
                              kernel_initializer=kernel_initializer,
                              use_bias=use_bias,
                              activation=activation,
                              padding=padding)(x)
            blocks.append(x)

        x = layers.Concatenate(axis=slice_axis)(blocks)
        return x

    return layer


def expand_dims(x, channels_axis):
    if channels_axis == 3:
        return x[:, None, None, :]
    elif channels_axis == 1:
        return x[:, :, None, None]
    else:
        raise ValueError("Slice axis should be in (1, 3), got {}.".format(channels_axis))


def ChannelSE(reduction=16, **kwargs):
    """
    Squeeze and Excitation block, reimplementation inspired by
        https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py

    Args:
        reduction: channels squeeze factor

    """
    channels_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor):
        # get number of channels/filters
        channels = backend.int_shape(input_tensor)[channels_axis]

        x = input_tensor

        # squeeze and excitation block in PyTorch style with
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Lambda(expand_dims, arguments={'channels_axis': channels_axis})(x)
        x = layers.Conv2D(channels // reduction, (1, 1), kernel_initializer='he_uniform')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(channels, (1, 1), kernel_initializer='he_uniform')(x)
        x = layers.Activation('sigmoid')(x)

        # apply attention
        x = layers.Multiply()([input_tensor, x])

        return x

    return layer


# -------------------------------------------------------------------------
#   Residual blocks
# -------------------------------------------------------------------------

def SEResNetBottleneck(filters, reduction=16, strides=1, **kwargs):
    bn_params = get_bn_params()

    def layer(input_tensor):
        x = input_tensor
        residual = input_tensor

        # bottleneck
        x = layers.Conv2D(filters // 4, (1, 1), kernel_initializer='he_uniform',
                          strides=strides, use_bias=False)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.ZeroPadding2D(1)(x)
        x = layers.Conv2D(filters // 4, (3, 3),
                          kernel_initializer='he_uniform', use_bias=False)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = layers.BatchNormalization(**bn_params)(x)

        #  if number of filters or spatial dimensions changed
        #  make same manipulations with residual connection
        x_channels = get_num_channels(x)
        r_channels = get_num_channels(residual)

        if strides != 1 or x_channels != r_channels:
            residual = layers.Conv2D(x_channels, (1, 1), strides=strides,
                                     kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = layers.BatchNormalization(**bn_params)(residual)

        # apply attention module
        x = ChannelSE(reduction=reduction, **kwargs)(x)

        # add residual connection
        x = layers.Add()([x, residual])

        x = layers.Activation('relu')(x)

        return x

    return layer


def SEResNeXtBottleneck(filters, reduction=16, strides=1, groups=32, base_width=4, **kwargs):
    bn_params = get_bn_params()

    def layer(input_tensor):
        x = input_tensor
        residual = input_tensor

        width = (filters // 4) * base_width * groups // 64

        # bottleneck
        x = layers.Conv2D(width, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.ZeroPadding2D(1)(x)
        x = GroupConv2D(width, (3, 3), strides=strides, groups=groups,
                        kernel_initializer='he_uniform', use_bias=False, **kwargs)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = layers.BatchNormalization(**bn_params)(x)

        #  if number of filters or spatial dimensions changed
        #  make same manipulations with residual connection
        x_channels = get_num_channels(x)
        r_channels = get_num_channels(residual)

        if strides != 1 or x_channels != r_channels:
            residual = layers.Conv2D(x_channels, (1, 1), strides=strides,
                                     kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = layers.BatchNormalization(**bn_params)(residual)

        # apply attention module
        x = ChannelSE(reduction=reduction, **kwargs)(x)

        # add residual connection
        x = layers.Add()([x, residual])

        x = layers.Activation('relu')(x)

        return x

    return layer


def SEBottleneck(filters, reduction=16, strides=1, groups=64, is_first=False, **kwargs):
    bn_params = get_bn_params()

    if is_first:
        downsample_kernel_size = (1, 1)
        padding = False
    else:
        downsample_kernel_size = (3, 3)
        padding = True

    def layer(input_tensor):

        x = input_tensor
        residual = input_tensor

        # bottleneck
        x = layers.Conv2D(filters // 2, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.ZeroPadding2D(1)(x)
        x = GroupConv2D(filters, (3, 3), strides=strides, groups=groups,
                        kernel_initializer='he_uniform', use_bias=False, **kwargs)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = layers.BatchNormalization(**bn_params)(x)

        #  if number of filters or spatial dimensions changed
        #  make same manipulations with residual connection
        x_channels = get_num_channels(x)
        r_channels = get_num_channels(residual)

        if strides != 1 or x_channels != r_channels:
            if padding:
                residual = layers.ZeroPadding2D(1)(residual)
            residual = layers.Conv2D(x_channels, downsample_kernel_size, strides=strides,
                                     kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = layers.BatchNormalization(**bn_params)(residual)

        # apply attention module
        x = ChannelSE(reduction=reduction, **kwargs)(x)

        # add residual connection
        x = layers.Add()([x, residual])

        x = layers.Activation('relu')(x)

        return x

    return layer


# -------------------------------------------------------------------------
#   SeNet builder
# -------------------------------------------------------------------------


def SENet(
        model_params,
        input_tensor=None,
        input_shape=None,
        include_top=True,
        classes=1000,
        weights='imagenet',
        **kwargs
):
    """Instantiates the ResNet, SEResNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Args:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if not (weights in {'imagenet', None} or file_io.file_exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    residual_block = model_params.residual_block
    init_filters = model_params.init_filters
    bn_params = get_bn_params()

    # define input
    if input_tensor is None:
        input = layers.Input(shape=input_shape, name='input')
    else:
        if not backend.is_keras_tensor(input_tensor):
            input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            input = input_tensor

    x = input

    if model_params.input_3x3:

        x = layers.ZeroPadding2D(1)(x)
        x = layers.Conv2D(init_filters, (3, 3), strides=2,
                          use_bias=False, kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.ZeroPadding2D(1)(x)
        x = layers.Conv2D(init_filters, (3, 3), use_bias=False,
                          kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.ZeroPadding2D(1)(x)
        x = layers.Conv2D(init_filters * 2, (3, 3), use_bias=False,
                          kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

    else:
        x = layers.ZeroPadding2D(3)(x)
        x = layers.Conv2D(init_filters, (7, 7), strides=2, use_bias=False,
                          kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

    x = layers.ZeroPadding2D(1)(x)
    x = layers.MaxPooling2D((3, 3), strides=2)(x)

    # body of resnet
    filters = model_params.init_filters * 2
    for i, stage in enumerate(model_params.repetitions):

        # increase number of filters with each stage
        filters *= 2

        for j in range(stage):

            # decrease spatial dimensions for each stage (except first, because we have maxpool before)
            if i == 0 and j == 0:
                x = residual_block(filters, reduction=model_params.reduction,
                                   strides=1, groups=model_params.groups, is_first=True, **kwargs)(x)

            elif i != 0 and j == 0:
                x = residual_block(filters, reduction=model_params.reduction,
                                   strides=2, groups=model_params.groups, **kwargs)(x)
            else:
                x = residual_block(filters, reduction=model_params.reduction,
                                   strides=1, groups=model_params.groups, **kwargs)(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        if model_params.dropout is not None:
            x = layers.Dropout(model_params.dropout)(x)
        x = layers.Dense(classes)(x)
        x = layers.Activation('softmax', name='output')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = input

    model_name = model_params.model_name
    model = models.Model(inputs, x, name=model_name)

    if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
        if include_top:
            file_name = model_name + '_imagenet_1000.h5'
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = model_name + '_imagenet_1000_no_top.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = data_utils.get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir='models',
            file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


# -------------------------------------------------------------------------
#   SE Residual Models
# -------------------------------------------------------------------------

MODELS_PARAMS = {
    'seresnet50': ModelParams(
        'seresnet50', repetitions=(3, 4, 6, 3), residual_block=SEResNetBottleneck,
        groups=1, reduction=16, init_filters=64, input_3x3=False, dropout=None,
    ),

    'seresnet101': ModelParams(
        'seresnet101', repetitions=(3, 4, 23, 3), residual_block=SEResNetBottleneck,
        groups=1, reduction=16, init_filters=64, input_3x3=False, dropout=None,
    ),

    'seresnet152': ModelParams(
        'seresnet152', repetitions=(3, 8, 36, 3), residual_block=SEResNetBottleneck,
        groups=1, reduction=16, init_filters=64, input_3x3=False, dropout=None,
    ),

    'seresnext50': ModelParams(
        'seresnext50', repetitions=(3, 4, 6, 3), residual_block=SEResNeXtBottleneck,
        groups=32, reduction=16, init_filters=64, input_3x3=False, dropout=None,
    ),

    'seresnext101': ModelParams(
        'seresnext101', repetitions=(3, 4, 23, 3), residual_block=SEResNeXtBottleneck,
        groups=32, reduction=16, init_filters=64, input_3x3=False, dropout=None,
    ),

    'senet154': ModelParams(
        'senet154', repetitions=(3, 8, 36, 3), residual_block=SEBottleneck,
        groups=64, reduction=16, init_filters=64, input_3x3=True, dropout=0.2,
    ),
}


def SEResNet50(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return SENet(
        MODELS_PARAMS['seresnet50'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def SEResNet101(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return SENet(
        MODELS_PARAMS['seresnet101'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def SEResNet152(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return SENet(
        MODELS_PARAMS['seresnet152'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def SEResNeXt50(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return SENet(
        MODELS_PARAMS['seresnext50'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def SEResNeXt101(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return SENet(
        MODELS_PARAMS['seresnext101'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def SENet154(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return SENet(
        MODELS_PARAMS['senet154'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )

preprocess_input = ImageNetNormalization(mode="torch", name="senet_preprocess")


setattr(SEResNet50, '__doc__', SENet.__doc__)
setattr(SEResNet101, '__doc__', SENet.__doc__)
setattr(SEResNet152, '__doc__', SENet.__doc__)
setattr(SEResNeXt50, '__doc__', SENet.__doc__)
setattr(SEResNeXt101, '__doc__', SENet.__doc__)
setattr(SENet154, '__doc__', SENet.__doc__)
