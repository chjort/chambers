import datetime
import json
import os

import faiss
import numpy as np
import tensorflow as tf

from chambers.models.base import set_predict_return_y
from chambers.utils.data import batch_predict_pairs
from chambers.utils.ranking import rank_labels


class FeatureRankingMetricCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        dataset: tf.data.Dataset,
        metric_funcs,
        feature_dim=None,
        name="ranking_metrics",
        use_gpu=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.metric_funcs = metric_funcs
        self.name = name
        self.feature_dim = feature_dim
        self.index = None
        self.use_gpu = use_gpu
        self._supports_tf_logs = True

    def set_model(self, model):
        if self.model is None:
            self.model = set_predict_return_y(model)

    def on_train_begin(self, logs=None):
        self.index = self._build_index()

    def on_epoch_end(self, epoch, logs=None):
        features, labels = self.model.predict(self.dataset)

        labels = labels.astype(int).reshape([-1])
        self.index.add_with_ids(features, labels)
        binary_ranking = self._compute_binary_ranking(
            features, labels, k=1001, remove_top1=True
        )

        for i, metric_fn in enumerate(self.metric_funcs):
            metric_name = "{}".format(metric_fn.__name__)
            metric_score = metric_fn(binary_ranking)
            logs[metric_name] = metric_score

        self.index.reset()

    def _compute_binary_ranking(self, features, labels, k=1000, remove_top1=True):
        _, knn_labels = self.index.search(features, k=k)
        if remove_top1:
            knn_labels = knn_labels[:, 1:]
        binary = np.equal(knn_labels, np.expand_dims(labels, 1)).astype(np.float32)
        return binary

    def _build_index(self):
        if self.feature_dim is None:
            model_output_dim = self.model.output_shape[-1]
            if model_output_dim is None:
                raise ValueError(
                    "Can not determine feature dimension from model output shape. Provide the 'feature_dim' argument."
                )
            self.feature_dim = model_output_dim

        INDEX_KEY = "IDMap,Flat"
        index = faiss.index_factory(self.feature_dim, INDEX_KEY)

        if self.use_gpu:
            gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

        return index


class PairedRankingMetricCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model,
        dataset: tf.data.Dataset,
        metric_funcs,
        encoder=None,
        batch_size=10,
        dataset_len=None,
        remove_top1=False,
        verbose=True,
        name="ranking_metrics",
    ):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.dataset = dataset
        self.metric_funcs = metric_funcs
        self.batch_size = batch_size
        self.dataset_len = dataset_len
        self.remove_top1 = remove_top1
        self.verbose = verbose
        self.name = name
        self._supports_tf_logs = True

        self.model = set_predict_return_y(model)
        if self.encoder is not None:
            self.encoder = set_predict_return_y(encoder)

    def set_model(self, model):
        if self.model is None:
            self.model = set_predict_return_y(model)

    def on_epoch_end(self, epoch, logs=None):
        if self.encoder is not None:
            qz, yq = self.encoder.predict(self.dataset)
            nq = len(qz)
        else:
            qz = self.dataset
            yq = None
            nq = self.dataset_len

        z, y = batch_predict_pairs(
            model=self.model,
            q=qz,
            bq=self.batch_size,
            yq=yq,
            nq=nq,
            verbose=self.verbose,
        )
        yqz, ycz = y
        y = tf.cast(tf.equal(yqz, tf.transpose(ycz)), tf.int32)

        binary_ranking, index_ranking = rank_labels(y, z, remove_top1=self.remove_top1)

        for i, metric_fn in enumerate(self.metric_funcs):
            metric_name = "{}".format(metric_fn.__name__)
            metric_score = metric_fn(binary_ranking)
            logs[metric_name] = metric_score


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
