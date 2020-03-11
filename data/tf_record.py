import tensorflow as tf

_SERIALIZE_TENSOR_DESCRIPTION = {
    "x": tf.io.FixedLenFeature([], tf.string),
    "y": tf.io.FixedLenFeature([], tf.int64),
    "dtype": tf.io.FixedLenFeature([], tf.string)
}


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_tensor_example(x, y):
    feature = {
        "x": _bytes_feature(tf.io.serialize_tensor(x)),
        "y": _int_feature(y),
        "dtype": _bytes_feature(tf.cast(x.dtype.name, tf.string)),
    }
    sample_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return sample_proto.SerializeToString()


def batch_deserialize_tensor_example(x):
    tensor_example = tf.io.parse_example(x, _SERIALIZE_TENSOR_DESCRIPTION)
    dtype_name = tensor_example["dtype"][0].numpy().decode("utf-8")  # get dtype from first example in batch
    dtype = tf.as_dtype(dtype_name)

    @tf.function
    def _parse_tensor_dtype(x):
        return tf.io.parse_tensor(x, out_type=dtype)

    x = tf.map_fn(_parse_tensor_dtype, tensor_example["x"], dtype=dtype,
                  parallel_iterations=10, infer_shape=False)
    y = tensor_example["y"]
    return x, y


def batch_deserialize_tensor_example_float32(x):
    tensor_example = tf.io.parse_example(x, _SERIALIZE_TENSOR_DESCRIPTION)

    @tf.function
    def _parse_tensor_float32(x):
        return tf.io.parse_tensor(x, out_type=tf.float32)

    x = tf.map_fn(_parse_tensor_float32, tensor_example["x"], dtype=tf.float32,
                  parallel_iterations=10, infer_shape=False)
    y = tensor_example["y"]
    return x, y


def deserialize_tensor_example(x):
    tensor_example = tf.io.parse_single_example(x, _SERIALIZE_TENSOR_DESCRIPTION)
    dtype_name = tensor_example["dtype"].numpy().decode("utf-8")  # get dtype from first example in batch
    dtype = tf.as_dtype(dtype_name)

    x = tf.io.parse_tensor(tensor_example["x"], out_type=dtype)
    y = tensor_example["y"]
    return x, y


def deserialize_tensor_example_float32(x):
    tensor_example = tf.io.parse_single_example(x, _SERIALIZE_TENSOR_DESCRIPTION)
    x = tf.io.parse_tensor(tensor_example["x"], out_type=tf.float32)
    y = tensor_example["y"]
    return x, y
