import tensorflow as tf


class RankingAccuracy:
    def __call__(self, binary_ranking):
        return ranking_accuracy(binary_ranking)

    @property
    def __name__(self):
        return "r_acc"


class RecallAtK:
    def __init__(self, k: int):
        self.k = k

    def __call__(self, binary_ranking):
        return recall_at_k(binary_ranking, self.k)

    @property
    def __name__(self):
        return "r@{}".format(self.k)


class PrecisionAtK:
    def __init__(self, k: int):
        self.k = k

    def __call__(self, binary_ranking):
        return precision_at_k(binary_ranking, self.k)

    @property
    def __name__(self):
        return "p@{}".format(self.k)


class MeanAveragePrecisionAtK:
    def __init__(self, k: int):
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
    accs = tf.TensorArray(
        dtype=tf.float32, size=nrows, element_shape=tf.TensorShape(dims=())
    )
    for i in tf.range(nrows):
        ki = tf.slice(n_true_pos, [i], [1])[0]
        n_pred_pos = tf.reduce_sum(tf.slice(binary_ranking, [i, 0], [1, ki]))
        acc = n_pred_pos / tf.cast(ki, tf.float32)
        accs = accs.write(i, acc)
    return tf.reduce_mean(accs.stack())


def recall_at_k(binary_ranking, k: int):
    return tf.reduce_mean(
        tf.cast(tf.reduce_sum(binary_ranking[:, :k], axis=1) > 0, tf.float32)
    )


def precision_at_k(binary_ranking, k: int):
    precisions_at_k = tf.reduce_sum(binary_ranking[:, :k], axis=1) / k
    return tf.reduce_mean(precisions_at_k)


def mean_average_precision(binary_ranking, k: int = None):
    if k is None or k > tf.shape(binary_ranking)[1]:
        k = tf.shape(binary_ranking)[1]

    binary_ranking = binary_ranking[:, :k]
    n_positive_predictions = tf.reduce_sum(binary_ranking, axis=1)
    rank_out_of_k = tf.range(1, k + 1, dtype=tf.float32)

    cumulative_positive_at_k = tf.cumsum(binary_ranking, axis=1) * binary_ranking
    k_precisions = tf.math.divide_no_nan(cumulative_positive_at_k, rank_out_of_k)
    average_precision_at_k = tf.math.divide_no_nan(
        tf.reduce_sum(k_precisions, axis=1), n_positive_predictions
    )
    return tf.reduce_mean(average_precision_at_k)
