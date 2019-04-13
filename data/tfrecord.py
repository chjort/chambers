import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def raw_image_bytes(img, label):
    """
    Takes an image path and a image label path and returns the image bytes for each
    :param img: image
    :param label: label image
    :return: image bytes, label bytes
    """
    img_bytes = tf.read_file(img)
    label_bytes = tf.read_file(label)
    return img_bytes, label_bytes


def serialize_example(img, label):
    """
    Creates a tf.Example message ready to be written to a file.
    :param img: image bytes
    :param label: label bytes
    :return: string. serialized example
    """

    feature = {
        'image': _bytes_feature(img),
        'label': _bytes_feature(label)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(img, label):
    """
    Wrapper for the "serialize_example" as a tensorflow function
    :param img: image bytes
    :param label: label bytes
    :return: string. serialized example
    """
    tf_string = tf.py_func(
        serialize_example,
        (img, label),
        tf.string
    )
    return tf.reshape(tf_string, ())


def set_to_tfrecord(filename, x, y):
    tf_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    tf_dataset = tf_dataset.map(raw_image_bytes)
    tf_dataset = tf_dataset.map(tf_serialize_example)

    writer = tf.data.experimental.TFRecordWriter(filename)
    with tf.Session() as sess:
        sess.run(writer.write(tf_dataset))


# Read TFRecord
def read_feature(example):
    feature_description = {
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.string)
    }
    return tf.parse_single_example(example, feature_description)