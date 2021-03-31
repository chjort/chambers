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
            raise ValueError("Unknown " + module_name + ":" + identifier)
        if inspect.isclass(obj):
            obj = obj(**kwargs)
        elif callable(obj):
            obj = obj(**kwargs)
        return obj

    else:
        raise ValueError(
            "Could not interpret serialized " + module_name + ": " + identifier
        )


def use_mixed_precision(dtype="mixed_float16", set_epsilon=False):
    policy = tf.keras.mixed_precision.experimental.Policy(name=dtype)
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print("Computation dtype:", policy.compute_dtype)
    print("Variable dtype:", policy.variable_dtype)
    if dtype.endswith("16") and set_epsilon:
        eps = 1e-4
        tf.keras.backend.set_epsilon(eps)
        print("Backend epsilon:", eps)


def set_random_seed(seed: int):
    # Python seeds
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Tensorflow seeds
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(seed)


def download_image(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3"
    }
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
        if layer_type == "Model":
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
    non_trainable_count = np.sum(
        [K.count_params(p) for p in model.non_trainable_weights]
    )

    number_size = 4.0
    if K.floatx() == "float16":
        number_size = 2.0
    if K.floatx() == "float64":
        number_size = 8.0

    total_memory = number_size * (
        batch_size * shapes_mem_count + trainable_count + non_trainable_count
    )
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


class ProgressBar:
    def __init__(self, total, cols=30):
        self.total = total
        self.cols = cols
        self._steps = tf.Variable(0, dtype=tf.int32)
        self._start_time = tf.timestamp()

    def _step_string(self):
        p_complete = self._steps / self.total

        n_complete = tf.cast(tf.math.floor(p_complete * self.cols), tf.int32)
        n_remaining = tf.cast(self.cols - n_complete, tf.int32)
        n_current = tf.cast(n_remaining > 0, tf.int32)
        n_remaining = n_remaining - n_current

        completed = tf.strings.reduce_join(tf.repeat("=", n_complete, axis=0))
        current = tf.strings.reduce_join(tf.repeat(">", n_current, axis=0))
        remaining = tf.strings.reduce_join(tf.repeat(".", n_remaining, axis=0))
        s = tf.strings.reduce_join(["[", completed, current, remaining, "]"])

        frac = tf.strings.reduce_join(
            [tf.as_string(self._steps), "/", tf.as_string(self.total)]
        )
        s = tf.strings.reduce_join(["\r", frac, " ", s])
        return s

    def _time_string(self):
        elapsed = tf.timestamp() - self._start_time
        time_per_step = elapsed / tf.cast(self._steps, tf.float64)

        # elapsed = self._num_to_string(elapsed, 2)
        time_per_step = self._num_to_string(time_per_step, 2)

        s = tf.strings.reduce_join([" - ", time_per_step, "s/step"])
        return s

    @staticmethod
    def _num_to_string(num, decimals=2):
        num_str = tf.as_string(num)
        if num.dtype.is_floating:
            len_ = tf.strings.length(num_str)
            decimals_to_remove = tf.maximum(
                0, 6 - decimals
            )  # float string has 6 decimals by default.
            num_str = tf.strings.substr(num_str, 0, len_ - decimals_to_remove)

        return num_str

    @staticmethod
    def _print(s):
        tf.print(s, end="")

    def _report_progress(self):
        ss = self._step_string()
        ts = self._time_string()
        self._print(ss)
        self._print(ts)

    def update(self, n):
        n = tf.cast(n, tf.int32)
        self._steps.assign(n)
        self._report_progress()

    def add(self, n):
        n = tf.cast(n, tf.int32)
        self._steps.assign_add(n)
        self._report_progress()
