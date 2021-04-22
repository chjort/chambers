import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
import types


# def MakeClassFromInstance(instance):
#     from copy import deepcopy
#     copy = deepcopy(instance.__dict__)
#     InstanceFactory = type('InstanceFactory', (instance.__class__,), {})
#     InstanceFactory.__init__ = lambda self, *args, **kwargs: self.__dict__.update(copy)
#     return InstanceFactory


class BaseModel(tf.keras.Model):
    @classmethod
    def from_model(cls, model):
        return cls(inputs=model.inputs, outputs=model.outputs, name=model.name)


class PredictReturnYModel(BaseModel):
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


def set_predict_return_y(model):
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

    model.predict_step = types.MethodType(predict_step, model)
    return model
