import tensorflow as tf


def _float_feature(value):
    """Returns a float_list from a float / double."""
    if tf.rank(value) == 0:
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if tf.rank(value) == 0:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Returns a bytes_list from a byte value."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    if tf.rank(value) == 0:
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


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
        shape = tf.shape(t)

        if t.shape != ():  # if tensor is not a scalar, we have to serialize it.
            t = tf.io.serialize_tensor(t)
            feature[name + "_serialized"] = _int_feature(1)
        else:
            feature[name + "_serialized"] = _int_feature(0)

        if t.dtype.is_floating:
            raw_feature = _float_feature(t)
        elif t.dtype.is_integer:
            raw_feature = _int_feature(t)
        elif t.dtype == tf.string:
            raw_feature = _bytes_feature(t)
        else:
            raise ValueError("Invalid dtype {}.".format(t.dtype))

        feature[name + "_raw"] = raw_feature
        feature[name + "_dtype"] = _int_feature(dtype.as_datatype_enum)
        feature[name + "_shape"] = _int_feature(shape)

    return feature


def _make_description(feature):
    tensors_ids = _get_tensor_ids(feature)

    description = {}
    for tn in tensors_ids:
        is_serialized = feature[tn + "_serialized"].int64_list.value[0]
        dtype = feature[tn + "_dtype"].int64_list.value[0]
        shape = feature[tn + "_shape"].int64_list.value

        description[tn + "_serialized"] = tf.io.FixedLenFeature([], tf.int64)
        if is_serialized:
            description[tn + "_raw"] = tf.io.FixedLenFeature([], tf.string)
        else:
            description[tn + "_raw"] = tf.io.FixedLenFeature([], tf.as_dtype(dtype))

        description[tn + "_dtype"] = tf.io.FixedLenFeature([], tf.int64)
        description[tn + "_shape"] = tf.io.FixedLenFeature([len(shape)], tf.int64)

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


def _make_feature_deserialize_fn(feature, set_shape=False, set_size=False):
    description = _make_description(feature)
    tensor_ids = _get_tensor_ids(feature)
    is_serialized = [
        feature[tn + "_serialized"].int64_list.value[0] for tn in tensor_ids
    ]
    dtypes = [
        tf.as_dtype(feature[tn + "_dtype"].int64_list.value[0]) for tn in tensor_ids
    ]
    if set_shape:
        shapes = [feature[tn + "_shape"].int64_list.value for tn in tensor_ids]
    elif set_size:
        shapes = [
            [None] * len(feature[tn + "_shape"].int64_list.value) for tn in tensor_ids
        ]

    def deserialize_fn(x):
        tensor_example = tf.io.parse_example(x, description)
        tensors = []
        for i, tn in enumerate(tensor_ids):
            name = tn + "_raw"
            serialized = is_serialized[i]
            dtype = dtypes[i]

            t = tensor_example[name]
            if serialized:
                t = tf.io.parse_tensor(t, out_type=dtype)

            if set_shape or set_size:
                shape = shapes[i]
                t.set_shape(shape)

            tensors.append(t)

        if len(tensors) == 1:
            tensors = tensors[0]
        else:
            tensors = tuple(tensors)

        return tensors

    return deserialize_fn


def make_dataset_deserialize_fn(dataset, set_shape=False, set_size=False) -> callable:
    sample = next(iter(dataset))

    example_features = tf.train.Example.FromString(sample.numpy())
    feature = example_features.features.feature
    return _make_feature_deserialize_fn(feature, set_shape=set_shape, set_size=set_size)


def dataset_to_tfrecord(dataset, path):
    dataset = dataset.map(serialize_example)

    writer = tf.data.experimental.TFRecordWriter(path)
    writer.write(dataset)


def tfrecord_to_dataset(paths, set_shape=False, set_size=False) -> tf.data.Dataset:
    td = tf.data.TFRecordDataset(paths)
    td = td.map(make_dataset_deserialize_fn(td, set_shape=set_shape, set_size=set_size))
    return td
