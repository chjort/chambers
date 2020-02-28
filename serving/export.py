import os

import tensorflow as tf


def export(path, model, preprocessing_fn, input_signature, export_name=None, version=1):

    @tf.function(input_signature=input_signature)
    def serving(input_images):
        with tf.device("cpu:0"):
            img = tf.map_fn(preprocessing_fn, input_images, dtype=tf.float32)

        embeddings = model(img)
        return embeddings

    if export_name is None:
        export_name = model.name

    save_path = os.path.join(path, export_name, str(version))
    tf.saved_model.save(model, save_path, signatures=serving)


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

# %%
# @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="image_bytes")])
#     def serving(input_images):
#         def _preprocess(image):
#             channels = 3
#             img = tf.image.decode_image(image, channels, expand_animations=False)
#             img.set_shape([None, None, channels])
#             img = resize(img, 256, 256)
#             img = center_crop(img, 224, 224)
#             img = resnet_normalize(img)
#             return img
#
#         # Preprocess
#         with tf.device("cpu:0"):
#             img = tf.map_fn(_preprocess, input_images, dtype=tf.float32)
#
#         # Encode
#         embeddings = model(img)
#
#         return embeddings

# output_path = "/home/crr/workspace/sik_models/outputs/to_deploy"
# model_name = "BN_Inception_sig"
# version = "1"
# save_path = os.path.join(output_path, model_name, version)
# tf.saved_model.save(model, save_path, signatures=serving)
#
