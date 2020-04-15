import inspect
import os
import random

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
