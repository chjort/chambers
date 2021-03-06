import time

import faiss
import numpy as np
import tensorflow as tf

from .utils.ranking import score_matrix_to_binary_ranking


class GlobalRankingMetricCallbackFaiss(tf.keras.callbacks.Callback):
    def __init__(self, dataset: tf.data.Dataset, metric_funcs, feature_dim=None, name="ranking_metrics", use_gpu=False):
        super().__init__()
        self.dataset = dataset
        self.metric_funcs = metric_funcs
        self.name = name
        self.feature_dim = feature_dim
        self.index = None
        self.use_gpu = use_gpu

    def on_train_begin(self, logs=None):
        self.index = self._build_index()

    def on_epoch_end(self, epoch, logs=None):
        features, labels = self._extract_features(self.dataset, self.model)

        labels = labels.astype(int)
        self.index.add_with_ids(features, labels)
        binary_ranking = self._compute_binary_ranking(features, labels, k=1001, remove_top1=True)

        for i, metric_fn in enumerate(self.metric_funcs):
            metric_name = "{}".format(metric_fn.__name__)
            metric_score = metric_fn(binary_ranking).numpy()
            logs[metric_name] = metric_score
            print(" - {}:".format(metric_name), np.round(metric_score, 4), end="")

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

    @staticmethod
    def _extract_features(dataset, model):
        features = []
        labels = []
        for x, y in dataset:
            feat = model(x)
            features.append(feat)
            labels.append(y)

        features = tf.concat(features, axis=0)
        labels = tf.concat(labels, axis=0)
        return features.numpy(), labels.numpy()


class GlobalRankingMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset: tf.data.Dataset, metric_funcs, batch_size=None, name="ranking_metrics",
                 device="/device:CPU:0"):
        super().__init__()
        self.dataset = dataset
        self.metric_funcs = metric_funcs
        self.batch_size = batch_size
        self.name = name
        self.device = device

    def on_epoch_end(self, epoch, logs=None):
        features, labels = self._extract_features(self.dataset, self.model)
        n, d = features.shape.as_list()
        feature_iter = tf.data.Dataset.from_tensor_slices((features, labels))
        if self.batch_size is not None:
            feature_iter = feature_iter.batch(self.batch_size)
            n_batches = np.ceil(n / self.batch_size).astype(int)
        else:
            feature_iter = feature_iter.batch(n)
            n_batches = 1

        with tf.device(self.device):
            metric_scores = np.zeros((n_batches, len(self.metric_funcs)))
            start = time.time()
            for i, (x, y) in feature_iter.enumerate():
                score_mat = tf.matmul(x, tf.transpose(features))
                binary_ranking = score_matrix_to_binary_ranking(score_mat, y, labels, remove_top1=True)
                for j, metric_fn in enumerate(self.metric_funcs):
                    metric_scores[i, j] = metric_fn(binary_ranking).numpy()
            print("Total time metric scoring:", time.time() - start)
            for i, metric_fn in enumerate(self.metric_funcs):
                metric_name = "{}".format(metric_fn.__name__)
                metric_score = metric_scores[:, i].mean(axis=0)
                logs[metric_name] = metric_score
                print(" - {}:".format(metric_name), np.round(metric_score, 4), end="")
            print()

    @staticmethod
    def _extract_features(dataset, model):
        features = []
        labels = []
        for x, y in dataset:
            feat = model(x)
            features.append(feat)
            labels.append(y)

        features = tf.concat(features, axis=0)
        labels = tf.concat(labels, axis=0)
        return features.numpy(), labels.numpy()
