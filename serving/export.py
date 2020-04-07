import os

import tensorflow as tf


def export(path, model, preprocessing_fn, input_signature, version=None):

    @tf.function(input_signature=input_signature)
    def serving(input_images):
        with tf.device("cpu:0"):
            img = tf.map_fn(preprocessing_fn, input_images, dtype=tf.float32)

        embeddings = model(img)
        return embeddings

    if version is not None:
        path = os.path.join(path, str(version))

    tf.saved_model.save(model, path, signatures=serving)


class Server:
    def __init__(self, model, preprocessing_fn, input_signature):
        self.model = model
        self.preprocessing_fn = preprocessing_fn
        self.input_signature = input_signature

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="image_bytes")])
    def serving(self, input_images):
        # Preprocess
        with tf.device("cpu:0"):
            img = tf.map_fn(self.preprocessing_fn, input_images, dtype=tf.float32)

        # Encode
        embeddings = self.model(img)

        return embeddings

    def save(self, path, name=None, version=1):
        if name is None:
            name = self.model.name

        save_path = os.path.join(path, name, str(version))
        tf.saved_model.save(self.model, save_path, signatures=self.serving)
