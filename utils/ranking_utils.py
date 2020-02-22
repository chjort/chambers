import tensorflow as tf

from .tf_utils import arg_to_gather_nd


def score_matrix_to_binary_ranking(score_matrix, query_labels, all_labels, remove_top1=False):
    query_labels = tf.reshape(query_labels, [-1, 1])
    all_labels = tf.reshape(all_labels, [-1, 1])
    pair_signs = tf.cast(tf.equal(query_labels, tf.transpose(all_labels)), tf.float32)

    index_ranking = tf.argsort(score_matrix, axis=1, direction="DESCENDING")
    if remove_top1:
        index_ranking = index_ranking[:, 1:]

    gather_idx = arg_to_gather_nd(index_ranking)
    binary_ranking = tf.reshape(tf.gather_nd(pair_signs, gather_idx), index_ranking.shape)

    return binary_ranking
