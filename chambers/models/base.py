import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter


class PredictDataModel(tf.keras.Model):
    def predict_step(self, data):
        """The logic for one inference step.

        This method can be overridden to support custom inference logic.
        This method is called by `Model.make_predict_function`.

        This method should contain the mathematical logic for one step of inference.
        This typically includes the forward pass.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_predict_function`, which can also be overridden.

        Arguments:
          data: A nested structure of `Tensor`s.

        Returns:
          The result of one inference step, typically the output of calling the
          `Model` on data.
        """
        data = data_adapter.expand_1d(data)
        x, y, _ = data_adapter.unpack_x_y_sample_weight(data)

        if y is None:
            return self(x, training=False)

        return self(x, training=False), y
