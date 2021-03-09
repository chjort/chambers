import json
import os

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
        t_raw = tf.io.serialize_tensor(t)
        dtype = t.dtype
        shape = tf.shape(t)

        name = "t" + str(i)
        feature[name + "_raw"] = _bytes_feature(t_raw)
        feature[name + "_dtype"] = _int_feature(dtype.as_datatype_enum)
        feature[name + "_shape"] = _int_feature(shape)

    return feature


def _make_description(feature):
    tensors_ids = _get_tensor_ids(feature)

    description = {}
    for tn in tensors_ids:
        shape = feature[tn + "_shape"].int64_list.value

        description[tn + "_raw"] = tf.io.FixedLenFeature([], tf.string)
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


def serialize_to_example(*args):
    example = tf.py_function(_serialize_example, args, tf.string)
    return tf.reshape(example, [])


def _make_feature_deserialize_fn(feature, set_shape=False, set_dimension=False):
    description = _make_description(feature)
    tensor_ids = _get_tensor_ids(feature)
    dtypes = [
        tf.as_dtype(feature[tn + "_dtype"].int64_list.value[0]) for tn in tensor_ids
    ]
    if set_shape:
        shapes = [feature[tn + "_shape"].int64_list.value for tn in tensor_ids]
    elif set_dimension:
        shapes = [
            [None] * len(feature[tn + "_shape"].int64_list.value) for tn in tensor_ids
        ]

    def deserialize_fn(x):
        tensor_example = tf.io.parse_example(x, description)
        tensors = []
        for i, tn in enumerate(tensor_ids):
            name = tn + "_raw"
            dtype = dtypes[i]

            t = tensor_example[name]
            t = tf.io.parse_tensor(t, out_type=dtype)

            if set_shape or set_dimension:
                shape = shapes[i]
                t.set_shape(shape)

            tensors.append(t)

        if len(tensors) == 1:
            tensors = tensors[0]
        else:
            tensors = tuple(tensors)

        return tensors

    return deserialize_fn


def make_dataset_deserialize_fn(
    dataset, set_shape=False, set_dimension=False
) -> callable:
    sample = next(iter(dataset))

    example_features = tf.train.Example.FromString(sample.numpy())
    feature = example_features.features.feature
    return _make_feature_deserialize_fn(
        feature, set_shape=set_shape, set_dimension=set_dimension
    )


def dataset_to_tfrecord(dataset, path):
    dataset = dataset.map(serialize_to_example)

    writer = tf.data.experimental.TFRecordWriter(path)
    writer.write(dataset)


def tfrecord_to_dataset(paths, set_shape=True, set_dimension=False) -> tf.data.Dataset:
    td = tf.data.TFRecordDataset(paths)
    td = td.map(
        make_dataset_deserialize_fn(
            td, set_shape=set_shape, set_dimension=set_dimension
        )
    )
    return td


# %%


def _tensor_spec_to_dict(tensor_spec):
    spec_dict = {
        "shape": list(tensor_spec.shape),
        "dtype": tensor_spec.dtype.as_datatype_enum,
        "name": tensor_spec.name,
    }
    return spec_dict


def _dict_to_tensor_spec(dictionary):
    shape = tf.TensorShape(dictionary["shape"])
    dtype = tf.as_dtype(dictionary["dtype"])
    name = dictionary["name"]
    tensor_spec = tf.TensorSpec(shape=shape, dtype=dtype, name=name)
    return tensor_spec


def _element_spec_to_json(element_spec):
    if isinstance(element_spec, tf.TensorSpec):
        return _tensor_spec_to_dict(element_spec)

    json_tensor_specs = []
    for tensor_spec in element_spec:
        json_tensor_specs.append(_element_spec_to_json(tensor_spec))

    return tuple(json_tensor_specs)


def _json_to_element_spec(json_element_spec):
    if isinstance(json_element_spec, dict):
        return _dict_to_tensor_spec(json_element_spec)

    tensor_specs = []
    for json_tensor_spec in json_element_spec:
        tensor_specs.append(_json_to_element_spec(json_tensor_spec))

    return tuple(tensor_specs)


def _dump_dataset_metadata(path, element_spec, enumerated):
    metadata = {
        "element_spec": _element_spec_to_json(element_spec),
        "enumerated": enumerated,
    }
    with open(path, "w") as f:
        json.dump(metadata, f)


def _load_dataset_metadata(path):
    with open(path, "r") as f:
        metadata = json.load(f)

    metadata["element_spec"] = _json_to_element_spec(metadata["element_spec"])
    return metadata


def save_dataset(dataset, path, n_files=1):
    if n_files > 1:

        def shard_func(*args):
            i, x = args
            return i % n_files

        dataset = dataset.enumerate()
        enumerated = True
    else:
        shard_func = None
        enumerated = False

    os.makedirs(path, exist_ok=True)
    _dump_dataset_metadata(
        path=os.path.join(path, "dataset.metadata"),
        element_spec=dataset.element_spec,
        enumerated=enumerated,
    )
    tf.data.experimental.save(dataset, path, shard_func=shard_func)


def load_dataset(path):
    metadata = _load_dataset_metadata(os.path.join(path, "dataset.metadata"))
    dataset = tf.data.experimental.load(path, element_spec=metadata["element_spec"])

    if metadata["enumerated"]:
        dataset = dataset.map(lambda i, x: x)

    return dataset
