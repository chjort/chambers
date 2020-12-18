import inspect
import os
import random
from urllib.request import urlopen, Request

import numpy as np
import tensorflow as tf


def deserialize_object(identifier, module_objects, module_name, **kwargs):
    if type(identifier) is str:
        obj = module_objects.get(identifier)
        if obj is None:
            raise ValueError('Unknown ' + module_name + ':' + identifier)
        if inspect.isclass(obj):
            obj = obj(**kwargs)
        elif callable(obj):
            obj = obj(**kwargs)
        return obj

    else:
        raise ValueError('Could not interpret serialized ' + module_name +
                         ': ' + identifier)


def set_random_seed(seed: int):
    # Python seeds
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Tensorflow seeds
    os.environ["TF_DETERMINISTIC_OPS"] = '1'
    tf.random.set_seed(seed)


def download_image(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3"}
    req = Request(url, headers=headers)

    img_file = urlopen(req)
    return img_file


def url_to_img_bytes(url):
    img_file = download_image(url)
    return img_file.read()


def url_to_img(url):
    img_bytes = url_to_img_bytes(url)
    return tf.io.decode_image(img_bytes)


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes
