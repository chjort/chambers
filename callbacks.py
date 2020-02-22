import numpy as np
import tensorflow as tf

from .utils.ranking_utils import score_matrix_to_binary_ranking


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
        features, labels = self._extract_features()
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
            for i, (x, y) in feature_iter.enumerate():
                score_mat = tf.matmul(x, tf.transpose(features))
                binary_ranking = score_matrix_to_binary_ranking(score_mat, y, labels, remove_top1=True)
                for j, metric_fn in enumerate(self.metric_funcs):
                    metric_scores[i, j] = metric_fn(binary_ranking).numpy()

            for i, metric_fn in enumerate(self.metric_funcs):
                metric_name = "{}".format(metric_fn.__name__)
                metric_score = metric_scores[:, i].mean(axis=0)
                logs[metric_name] = metric_score
                print(" - {}:".format(metric_name), np.round(metric_score, 4), end="")
            print()

    def _extract_features(self):
        features = []
        labels = []
        for x, y in self.dataset:
            feat = self.model(x)
            features.append(feat)
            labels.append(y)

        features = tf.concat(features, axis=0)
        labels = tf.concat(labels, axis=0)
        return features, labels
