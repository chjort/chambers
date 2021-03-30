from tensorflow.python.keras.applications.resnet import stack3, ResNet

from chambers.augmentations import ImageNetNormalization


def ResNeXt50(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              **kwargs):
    def stack_fn(x):
        x = stack3(x, 128, 3, stride1=1, name='conv2')
        x = stack3(x, 256, 4, name='conv3')
        x = stack3(x, 512, 6, name='conv4')
        x = stack3(x, 1024, 3, name='conv5')
        return x

    return ResNet(stack_fn, False, False, 'resnext50',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes,
                  **kwargs)


def ResNeXt101(include_top=True,
               weights='imagenet',
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               **kwargs):
    def stack_fn(x):
        x = stack3(x, 128, 3, stride1=1, name='conv2')
        x = stack3(x, 256, 4, name='conv3')
        x = stack3(x, 512, 23, name='conv4')
        x = stack3(x, 1024, 3, name='conv5')
        return x

    return ResNet(stack_fn, False, False, 'resnext101',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes,
                  **kwargs)


preprocess_input = ImageNetNormalization(mode="torch", name="resnext_preprocess")

setattr(ResNeXt50, '__doc__', ResNet.__doc__)
setattr(ResNeXt101, '__doc__', ResNet.__doc__)
