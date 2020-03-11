from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils


class Miner(ABC):
    def __init__(self, name=None):
        self.name = name

    def __call__(self, positive, negative):
        scope_name = self.name
        graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
            positive, negative)
        with K.name_scope(scope_name or self.__class__.__name__), graph_ctx:
            return self.call(positive, negative)

    @abstractmethod
    def call(self, positive, negative):
        pass


class MultiSimilarityMiner(Miner):
    def __init__(self, margin, name="multi_similarity_miner"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, positive, negative):
        pos_thresh = tf.reduce_max(negative, axis=1) + self.margin
        neg_thresh = tf.reduce_min(positive, axis=1) - self.margin

        mined_pos_mask = positive < tf.reshape(pos_thresh, [-1, 1])
        mined_neg_mask = negative > tf.reshape(neg_thresh, [-1, 1])

        mined_pos_mat = tf.ragged.boolean_mask(positive, mined_pos_mask)
        mined_neg_mat = tf.ragged.boolean_mask(negative, mined_neg_mask)

        return mined_pos_mat, mined_neg_mat
