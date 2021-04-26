import os

import faiss
import numpy as np
import tensorflow as tf

def extract_features(dataset, model):
    features = []
    labels = []
    for x, y in dataset:
        feat = model(x, training=False)
        features.append(feat)
        labels.append(y)

    features = tf.concat(features, axis=0)
    labels = tf.concat(labels, axis=0)
    return features.numpy(), labels.numpy()


class GlobalRankingMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset: tf.data.Dataset, metric_funcs, feature_dim=None, name="ranking_metrics", use_gpu=False):
        super().__init__()
        self.dataset = dataset
        self.metric_funcs = metric_funcs
        self.name = name
        self.feature_dim = feature_dim
        self.index = None
        self.use_gpu = use_gpu
        self._supports_tf_logs = True

    def on_train_begin(self, logs=None):
        self.index = self._build_index()

    def on_epoch_end(self, epoch, logs=None):
        features, labels = extract_features(self.dataset, self.model)

        labels = labels.astype(int)
        self.index.add_with_ids(features, labels)
        binary_ranking = self._compute_binary_ranking(features, labels, k=1001, remove_top1=True)

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
                    "Can not determine feature dimension from model output shape. Provide the 'feature_dim' argument.")
            self.feature_dim = model_output_dim

        INDEX_KEY = "IDMap,Flat"
        index = faiss.index_factory(self.feature_dim, INDEX_KEY)

        if self.use_gpu:
            gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

        return index


class ExperimentCallback(tf.keras.callbacks.CallbackList):
    def __init__(
        self,
        experiments_dir,
        checkpoint_monitor,
        checkpoint_mode="max",
        tensorboard_update_freq="epoch",
        tensorboard_write_graph=True,
    ):
        self.log_dir = os.path.join(experiments_dir, "logs")
        self.model_dir = os.path.join(experiments_dir, "model")
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        self.export_dir = os.path.join(self.model_dir, "export")

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

        super(ExperimentCallback, self).__init__(
            callbacks=callbacks, add_history=False, add_progbar=False
        )

    def on_train_begin(self, logs=None):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)
        self.model.save_weights(os.path.join(self.checkpoint_dir, "init.h5"))
        super(ExperimentCallback, self).on_train_begin(logs)

    def on_train_end(self, logs=None):
        self.model.save(os.path.join(self.model_dir, "save"), include_optimizer=True)
        super(ExperimentCallback, self).on_train_end(logs)