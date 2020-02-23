import tensorflow as tf


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