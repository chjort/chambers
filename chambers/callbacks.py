import datetime
import json
import os

import tensorflow as tf


class ExperimentCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        experiments_dir,
        checkpoint_monitor="val_loss",
        checkpoint_mode="auto",
        tensorboard_update_freq="epoch",
        tensorboard_write_graph=True,
        config_dump=None,
    ):
        super(ExperimentCallback, self).__init__()
        now_timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.experiment_dir = os.path.join(experiments_dir, now_timestamp)
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        self.model_dir = os.path.join(self.experiment_dir, "model")
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        self.export_dir = os.path.join(self.model_dir, "export")

        self.config_dump = config_dump

        csv_logger = tf.keras.callbacks.CSVLogger(
            filename=os.path.join(self.log_dir, "epoch_results.txt")
        )
        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                self.checkpoint_dir, "{epoch:02d}-{" + checkpoint_monitor + ":.5f}.h5"
            ),
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
            save_weights_only=True,
        )
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            update_freq=tensorboard_update_freq,
            profile_batch=0,
            write_graph=tensorboard_write_graph,
        )

        callbacks = [csv_logger, checkpointer, tensorboard]
        self._callback_list = tf.keras.callbacks.CallbackList(
            callbacks=callbacks, add_history=False, add_progbar=False
        )

    def set_params(self, params):
        self.params = params
        self._callback_list.set_params(params)

    def set_model(self, model):
        self.model = model
        self._callback_list.set_model(model)

    def on_batch_begin(self, batch, logs=None):
        self._callback_list.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        self._callback_list.on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._callback_list.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self._callback_list.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        self._callback_list.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self._callback_list.on_train_batch_end(batch, logs)

    def on_test_batch_begin(self, batch, logs=None):
        self._callback_list.on_test_batch_begin(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self._callback_list.on_test_batch_end(batch, logs)

    def on_predict_batch_begin(self, batch, logs=None):
        self._callback_list.on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        self._callback_list.on_predict_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)

        if self.config_dump is not None:
            with open(os.path.join(self.experiment_dir, "config_dump.json"), "w") as f:
                json.dump(self.config_dump, f)

        self.model.save_weights(os.path.join(self.checkpoint_dir, "init.h5"))
        self._callback_list.on_train_begin(logs)

    def on_train_end(self, logs=None):
        self.model.save(os.path.join(self.export_dir), include_optimizer=True)
        self._callback_list.on_train_end(logs)

    def on_test_begin(self, logs=None):
        self._callback_list.on_test_begin(logs)

    def on_test_end(self, logs=None):
        self._callback_list.on_test_end(logs)

    def on_predict_begin(self, logs=None):
        self._callback_list.on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        self._callback_list.on_predict_end(logs)
