import tensorflow as tf
from abc import ABC, abstractmethod

class Miner(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def mine(self, positive, negative):
        pass


class MultiSimilarityMiner(Miner):
    def __init__(self, margin, name="multi_similarity_miner"):
            super().__init__(name=name)
            self.margin = margin

    def mine(self, positive, negative):
        pos_thresh = tf.reduce_max(negative, axis=1) + self.margin
        neg_thresh = tf.reduce_min(positive, axis=1) - self.margin

        mined_pos_mask = positive < tf.reshape(pos_thresh, [-1, 1])
        mined_neg_mask = negative > tf.reshape(neg_thresh, [-1, 1])

        mined_pos_mat = tf.ragged.boolean_mask(positive, mined_pos_mask)
        mined_neg_mat = tf.ragged.boolean_mask(negative, mined_neg_mask)

        return mined_pos_mat, mined_neg_mat