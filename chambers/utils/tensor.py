from typing import List

import tensorflow as tf


def remove_indices(x, indices: List[int], axis=0):
    len_axis = tf.shape(x)[axis]

    mask_indices = tf.expand_dims(tf.convert_to_tensor(indices, dtype=tf.int32), -1)
    falses = tf.zeros_like(indices, dtype=tf.bool)
    mask = tf.ones([len_axis], dtype=tf.bool)
    mask = tf.tensor_scatter_nd_update(mask, mask_indices, falses)

    keep_indices = tf.range(len_axis)[mask]
    x = tf.gather(x, keep_indices, axis=axis)
    return x


def remove_diagonal(mat):
    n = tf.shape(mat)[0]
    m = tf.shape(mat)[1]

    diag_mask = tf.ones_like(mat, dtype=tf.bool)
    diag_mask = tf.linalg.set_diag(diag_mask, tf.cast(tf.zeros([n]), tf.bool))
    return tf.reshape(mat[diag_mask], [n, m - 1])


def arg_to_gather_nd(arg):
    n = tf.shape(arg)[0]

    row_index_ranking = tf.ones_like(arg) * tf.reshape(tf.range(n), [-1, 1])
    index_ranking = tf.stack([row_index_ranking, arg], axis=-1)
    gather_idx = tf.reshape(index_ranking, [-1, 2])
    return gather_idx
