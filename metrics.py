import tensorflow as tf


# from .utils.ranking_utils import score_matrix_to_binary_ranking
#
#
# class GlobalMeanAveragePrecision(tf.keras.metrics.Metric):
#     def __init__(self, max_samples, dim, name="ranking_metrics"):
#         super().__init__(name=name)
#         self.max_samples = max_samples
#         self.dim = dim
#         self.y_preds = self.add_weight(shape=(max_samples, dim), name="y_preds")
#         self.y_trues = self.add_weight(shape=(max_samples, 1), name="y_trues")
#         self.n_trues = self.add_weight(name="n_trues", dtype=tf.int64, initializer="zeros")
#         self.n_preds = self.add_weight(name="n_preds", dtype=tf.int64, initializer="zeros")
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         n_true = tf.cast(tf.shape(y_true)[0], tf.int64)
#         n_pred = tf.cast(tf.shape(y_pred)[0], tf.int64)
#
#         tf.print(n_true)
#         tf.print(n_pred)
#
#         true_current = self._get_current_trues()
#         true_fill = tf.zeros((self.max_samples - (self.n_trues + n_true), 1))
#         self.y_trues.assign(tf.concat([true_current, true_fill], axis=0))
#
#         pred_current = self._get_current_preds()
#         pred_fill = tf.zeros((self.max_samples - (self.n_preds + n_pred), self.dim))
#         self.y_preds.assign(tf.concat([pred_current, pred_fill], axis=0))
#
#         self.n_trues.assign_add(n_true)
#         self.n_preds.assign_add(n_pred)
#
#     def reset_states(self):
#         self.y_trues.assign(tf.zeros_like(self.y_trues))
#         self.y_preds.assign(tf.zeros_like(self.y_preds))
#
#     def result(self, k=10):
#         y_trues = self._get_current_trues()
#         y_preds = self._get_current_preds()
#
#         score_mat = tf.matmul(y_preds, tf.transpose(y_preds))
#         score_mat = score_matrix_to_binary_ranking(score_mat, y_trues, remove_diag=True)
#         return mean_average_precision(score_mat, k)
#
#     def _get_current_trues(self):
#         return tf.slice(self.y_trues, [0, 0], [self.n_trues, -1])
#
#     def _get_current_preds(self):
#         return tf.slice(self.y_preds, [0, 0], [self.n_preds, -1])


# class GlobalMeanAveragePrecision():
#     def __init__(self, dataset: tf.data.Dataset, model):
#         self.dataset = dataset
#         self.model = model
#         self.features = None
#         self.labels = None
#
#     def mean_average_precision(self, y_true, y_pred):
#         return 0.5
#
#     def _extract_features(self):
#         features = []
#         labels = []
#         for x, y in self.dataset:
#             feat = self.model(x)
#             features.append(feat)
#             labels.append(y)
#
#         self.features = tf.concat(features, axis=0)
#         self.labels = tf.concat(labels, axis=0)


class RankingAccuracy:

    def __call__(self, binary_ranking):
        return ranking_accuracy(binary_ranking)

    @property
    def __name__(self):
        return "r_acc"


class RecallAtK:
    def __init__(self, k):
        self.k = k

    def __call__(self, binary_ranking):
        return recall_at_k(binary_ranking, self.k)

    @property
    def __name__(self):
        return "r@{}".format(self.k)


class PrecisionAtK:
    def __init__(self, k):
        self.k = k

    def __call__(self, binary_ranking):
        return precision_at_k(binary_ranking, self.k)

    @property
    def __name__(self):
        return "p@{}".format(self.k)


class MeanAveragePrecisionAtK:
    def __init__(self, k):
        self.k = k

    def __call__(self, binary_ranking):
        return mean_average_precision(binary_ranking, self.k)

    @property
    def __name__(self):
        return "map@{}".format(self.k)


class MeanAveragePrecision:

    def __call__(self, binary_ranking):
        return mean_average_precision(binary_ranking, None)

    @property
    def __name__(self):
        return "map"


@tf.function
def ranking_accuracy(binary_ranking):
    n_true_pos = tf.cast(tf.reduce_sum(binary_ranking, axis=1), tf.int32)
    nrows = tf.shape(binary_ranking)[0]
    accs = tf.TensorArray(dtype=tf.float32, size=nrows, element_shape=tf.TensorShape(dims=()))
    for i in tf.range(nrows):
        ki = tf.slice(n_true_pos, [i], [1])[0]
        n_pred_pos = tf.reduce_sum(tf.slice(binary_ranking, [i, 0], [1, ki]))
        acc = n_pred_pos / tf.cast(ki, tf.float32)
        accs = accs.write(i, acc)
    return tf.reduce_mean(accs.stack())


def recall_at_k(binary_ranking, k):
    return tf.reduce_mean(tf.cast(tf.reduce_sum(binary_ranking[:, :k], axis=1) > 0, tf.float32))


def precision_at_k(binary_ranking, k):
    precisions_at_k = tf.reduce_sum(binary_ranking[:, :k], axis=1) / k
    return tf.reduce_mean(precisions_at_k)


# @tf.function
def mean_average_precision(binary_ranking, k=None):
    if k is None:
        k = tf.shape(binary_ranking)[1]

    binary_ranking = binary_ranking[:, :k]
    n_positive_predictions = tf.reduce_sum(binary_ranking, axis=1)
    rank_out_of_k = tf.range(1, k + 1, dtype=tf.float32)

    cumulative_positive_at_k = tf.cumsum(binary_ranking, axis=1) * binary_ranking
    k_precisions = tf.math.divide_no_nan(cumulative_positive_at_k, rank_out_of_k)
    average_precision_at_k = tf.math.divide_no_nan(tf.reduce_sum(k_precisions, axis=1), n_positive_predictions)
    return tf.reduce_mean(average_precision_at_k)
