import json
import os

import tensorflow as tf


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
