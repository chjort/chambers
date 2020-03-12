import abc
from typing import Tuple

import tensorflow as tf

from ..miners import MultiSimilarityMiner as _MSMiner


class PairBasedLoss(tf.keras.losses.Loss, abc.ABC):
    def __init__(self, miner=None, name="pair_based_loss"):
        """
        :param miner: The miner to use for sample mining
        :param name: Name of the loss function.
        """
        super().__init__(name=name)
        self.miner = miner

    def call(self, y_true, y_pred) -> tf.Tensor:
        """
        Computes the loss of embeddings based on the similarity between pairs.
        :param y_true: The class labels for the embeddings as a vector with shape [n]
        :param y_pred: The embeddings as a matrix with shape [n, m]
        :return: The loss as a scalar Tensor
        """
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)

        similarity_matrix = self.compute_similarity_matrix(y_pred)

        # split similarity matrix into similarites for positive pairs and similarities for negative pairs
        positive_pairs, negative_pairs = self.get_signed_pairs(similarity_matrix, y_true, ignore_diag=True)

        if self.miner is not None:
            # Mine for informative pairs
            positive_pairs, negative_pairs = self.miner(positive_pairs, negative_pairs)

        loss = self.compute_loss(positive_pairs, negative_pairs)
        return loss

    def compute_similarity_matrix(self, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the dot product similarity between all embedding pairs.

        :param y_pred: The embeddings as a Tensor with shape [n, embedding dimension].
        :return: The similarity scores as a 2D Tensor with shape [n, n]
        """
        return tf.matmul(y_pred, tf.transpose(y_pred))

    def get_signed_pairs(self, similarity_matrix: tf.Tensor, y_true: tf.Tensor, ignore_diag: bool = True) -> \
            Tuple[tf.RaggedTensor, tf.RaggedTensor]:
        """

        :param similarity_matrix: The similarity scores between the embeddings as A 2D Tensor of shape
                [n, n].
        :param y_true: The class labels for the embeddings as a Tensor with shape [n].
        :param ignore_diag: If True the diagonal pairs of the similarity matrix will be ignored.
        :return: Positive pairs and negative pairs as a tuple of 2D RaggedTensors each with shape [n, 0... n].
        """
        y_true = tf.reshape(y_true, [-1, 1])
        pos_pair_mask = tf.equal(y_true, tf.transpose(y_true))
        neg_pair_mask = tf.logical_not(pos_pair_mask)

        if ignore_diag:
            # ignore mirror pairs
            diag_len = tf.shape(similarity_matrix)[0]
            diag_val = tf.tile([False], [diag_len])
            pos_pair_mask = tf.linalg.set_diag(pos_pair_mask, diag_val)
            neg_pair_mask = tf.linalg.set_diag(neg_pair_mask, diag_val)

        # get similarities of positive pairs
        pos_mat = tf.RaggedTensor.from_row_lengths(values=similarity_matrix[pos_pair_mask],
                                                   row_lengths=tf.reduce_sum(tf.cast(pos_pair_mask, tf.int32), axis=1)
                                                   )

        # get similarities of negative pairs
        neg_mat = tf.RaggedTensor.from_row_lengths(values=similarity_matrix[neg_pair_mask],
                                                   row_lengths=tf.reduce_sum(tf.cast(neg_pair_mask, tf.int32), axis=1)
                                                   )

        return pos_mat, neg_mat

    @abc.abstractmethod
    def compute_loss(self, positive_pairs: tf.RaggedTensor, negative_pairs: tf.RaggedTensor):
        """
        Computes the total loss given the positive pair similarities and negative pair similarities.
        :param positive_pairs: Positive pairs as a 2D RaggedTensor with shape [n, 0... n].
        :param negative_pairs: Negative pairs as a 2D RaggedTensor with shape [n, 0... n].
        :return:
        """
        pass


class MultiSimilarityLoss(PairBasedLoss):
    """
    Multi-similarity loss

    References:

    [1] Wang, Xun et al. “Multi-Similarity Loss With General Pair Weighting for Deep Metric Learning.”
    2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2019): 5017-5025.
    https://arxiv.org/abs/1904.06627
    """

    def __init__(self, pos_scale=2.0, neg_scale=40.0, threshold=0.5, miner=_MSMiner(margin=0.1),
                 name="multi_similarity_loss"):
        super().__init__(miner=miner, name=name)
        self.pos_scale = pos_scale  # alpha
        self.neg_scale = neg_scale  # beta
        self.threshold = threshold  # lambda

    def compute_loss(self, positive_pairs, negative_pairs):
        pos_loss = tf.math.log(
            1 + tf.reduce_sum(tf.exp(-self.pos_scale * (positive_pairs - self.threshold)), axis=1)) / self.pos_scale
        neg_loss = tf.math.log(
            1 + tf.reduce_sum(tf.exp(self.neg_scale * (negative_pairs - self.threshold)), axis=1)) / self.neg_scale

        return pos_loss + neg_loss


class ContrastiveLoss(PairBasedLoss):
    def __init__(self, positive_margin=1., negative_margin=0.3, exponent=2, miner=None, name="contrastive_loss"):
        """

        :param positive_margin: The margin that the similarity of positive pairs at least should be to not contribute
        loss
        :param negative_margin: The margin that the similarity of negative pairs at most should be to not contribute
        loss
        :param exponent: The exponent which the losses are raised to the power of.
        """
        super().__init__(miner=miner, name=name)
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.exponent = exponent

    def compute_loss(self, positive_pairs: tf.RaggedTensor, negative_pairs: tf.RaggedTensor):
        # if positive pair similarity is lower than the positive margin, it contributes to the loss
        pos_pairs_loss = tf.pow(self.positive_margin - positive_pairs, self.exponent) / self.exponent
        pos_loss = tf.reduce_sum(pos_pairs_loss, axis=1)

        # if negative pair similarity is larger than the negative margin, it contributes to the loss
        neg_pairs_loss = tf.pow(tf.maximum(0, negative_pairs - self.negative_margin), self.exponent) / self.exponent
        neg_loss = tf.reduce_sum(neg_pairs_loss, axis=1)

        return pos_loss + neg_loss


class NTXentLoss(PairBasedLoss):
    def __init__(self, temperature, miner=None, name=None):
        super().__init__(miner=miner, name=name)
        self.temperature = temperature

    def compute_loss(self, positive_pairs: tf.RaggedTensor, negative_pairs: tf.RaggedTensor):

        # loss = tf.nn.softmax_cross_entropy_with_logits()
        raise NotImplementedError()

# class NewLoss(PairBasedLoss):
#     def __init__(self, miner=None, name=None):
#         super().__init__(miner=miner, name=name)
