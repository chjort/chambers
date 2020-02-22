import tensorflow as tf
from .miners import MultiSimilarityMiner as _MSMiner
import abc


class PairBasedLoss(tf.keras.losses.Loss):
    def __init__(self, name):
        super().__init__(name=name)

    @abc.abstractmethod
    def compute_similarity_matrix(self, y_true, y_pred):
        pass

    def get_signed_pairs(self, sim_mat, y_true, remove_positive_diag=True):
        y_true = tf.reshape(y_true, [-1, 1])
        pos_pair_mask = tf.equal(y_true, tf.transpose(y_true))
        neg_pair_mask = tf.logical_not(pos_pair_mask)

        if remove_positive_diag:
            # remove mirror pairs
            diag_len = tf.shape(sim_mat)[0]
            pos_pair_mask = tf.linalg.set_diag(pos_pair_mask, tf.tile([False], [diag_len]))

        # get similarities for positive pairs
        pos_mat = tf.RaggedTensor.from_row_lengths(values=sim_mat[pos_pair_mask],
                                                   row_lengths=tf.reduce_sum(tf.cast(pos_pair_mask, tf.int32), axis=1)
                                                   )

        # get similarities for negative pairs
        neg_mat = tf.RaggedTensor.from_row_lengths(values=sim_mat[neg_pair_mask],
                                                   row_lengths=tf.reduce_sum(tf.cast(neg_pair_mask, tf.int32), axis=1)
                                                   )

        return pos_mat, neg_mat


class MultiSimilarityLoss(PairBasedLoss):
    def __init__(self, pos_scale=2.0, neg_scale=40.0, threshold=0.5, miner=_MSMiner(margin=0.1), name="multi_similarity_loss"):
        super().__init__(name=name)
        self.pos_scale = pos_scale
        self.neg_scale = neg_scale
        self.threshold = threshold
        self.miner = miner

    def call(self, y_true, y_pred):
        sim_mat = self.compute_similarity_matrix(y_true, y_pred)

        # split similarity matrix into similarites for positive pairs and similarities for negative pairs
        pos_mat, neg_mat = self.get_signed_pairs(sim_mat, y_true, remove_positive_diag=True)

        if self.miner is not None:
            # Mine for informative pairs
            pos_mat, neg_mat = self.miner.mine(pos_mat, neg_mat)

        pos_loss = tf.math.log(1 + tf.reduce_sum(tf.exp(-self.pos_scale * (pos_mat - self.threshold)), axis=1)) / self.pos_scale
        neg_loss = tf.math.log(1 + tf.reduce_sum(tf.exp(self.neg_scale * (neg_mat - self.threshold)), axis=1)) / self.neg_scale

        ms_loss = pos_loss + neg_loss

        return tf.reduce_mean(ms_loss)

    def compute_similarity_matrix(self, y_true, y_pred):
        return tf.matmul(y_pred, tf.transpose(y_pred))