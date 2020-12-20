import tensorflow as tf


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a byte value."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_tensor_ids(dictionary):
    """ Extract the tensor Ids from dictionary which is either a feature or description"""
    tensor_ids = [k.split("_")[0] for k in dictionary.keys()]
    unique_tensor_ids = list(set(tensor_ids))
    return sorted(unique_tensor_ids)


def _make_feature(tensors):
    if not isinstance(tensors, (list, tuple)):
        tensors = (tensors,)

    feature = {}
    for i, t in enumerate(tensors):
        name = "t" + str(i)

        dtype = t.dtype

        if t.shape != ():
            t = tf.io.serialize_tensor(t)

        ftype = t.dtype
        if ftype.is_floating:
            raw_feature = _float_feature(t)
        elif ftype.is_integer:
            raw_feature = _int_feature(t)
        elif ftype == tf.string:
            raw_feature = _bytes_feature(t)
        else:
            raise ValueError("Invalid dtype {}.".format(ftype))

        feature[name + "_raw"] = raw_feature
        feature[name + "_dtype"] = _int_feature(dtype.as_datatype_enum)
        feature[name + "_ftype"] = _int_feature(ftype.as_datatype_enum)

    return feature


def _make_description(feature):
    tensors_ids = _get_tensor_ids(feature)

    description = {}
    for tn in tensors_ids:
        dtype = feature[tn + "_ftype"].int64_list.value[0]
        description[tn + "_raw"] = tf.io.FixedLenFeature([], tf.DType(dtype))

        description[tn + "_dtype"] = tf.io.FixedLenFeature([], tf.int64)
        description[tn + "_ftype"] = tf.io.FixedLenFeature([], tf.int64)

    return description


def _feature_to_example(feature):
    sample_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return sample_proto.SerializeToString()


def _serialize_example(*args):
    feature = _make_feature(args)
    example = _feature_to_example(feature)
    return example


def serialize_example(*args):
    example = tf.py_function(_serialize_example, args, tf.string)
    return tf.reshape(example, [])


def _make_feature_deserialize_fn(feature):
    description = _make_description(feature)
    tensor_ids = _get_tensor_ids(feature)
    dtypes = [
        tf.as_dtype(feature[tn + "_dtype"].int64_list.value[0]) for tn in tensor_ids
    ]
    ftypes = [
        tf.as_dtype(feature[tn + "_ftype"].int64_list.value[0]) for tn in tensor_ids
    ]
    do_convert = [dtype != ftype for dtype, ftype in zip(dtypes, ftypes)]

    def deserialize_fn(x):
        tensor_example = tf.io.parse_example(x, description)
        tensors = []
        for tn, dtype, convert in zip(tensor_ids, dtypes, do_convert):
            name = tn + "_raw"
            t = tensor_example[name]
            if convert:
                t = tf.io.parse_tensor(t, out_type=dtype)
            tensors.append(t)

        if len(tensors) == 1:
            tensors = tensors[0]
        else:
            tensors = tuple(tensors)

        return tensors

    return deserialize_fn


def make_dataset_deserialize_fn(dataset):
    sample = next(iter(dataset))

    example_features = tf.train.Example.FromString(sample.numpy())
    feature = example_features.features.feature
    return _make_feature_deserialize_fn(feature)
