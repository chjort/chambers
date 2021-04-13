import tensorflow as tf

from chambers.models.base import PredictReturnYModel
from chambers.models.bloodhound import batch_predict_pairs
from chambers.utils.ranking import rank_labels


class PairedRankingMetricCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model,
        dataset: tf.data.Dataset,
        metric_funcs,
        encoder=None,
        batch_size=10,
        dataset_len=None,
        remove_top1=False,
        verbose=True,
        name="ranking_metrics",
    ):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.dataset = dataset
        self.metric_funcs = metric_funcs
        self.batch_size = batch_size
        self.dataset_len = dataset_len
        self.remove_top1 = remove_top1
        self.verbose = verbose
        self.name = name
        self._supports_tf_logs = True

        self.model = PredictReturnYModel.from_model(model)
        if self.encoder is not None:
            self.encoder = PredictReturnYModel.from_model(encoder)

    def set_model(self, model):
        if self.model is None:
            self.model = PredictReturnYModel.from_model(model)

    def on_epoch_end(self, epoch, logs=None):
        if self.encoder is not None:
            qz, yq = self.encoder.predict(self.dataset)
            nq = len(qz)
        else:
            qz = self.dataset
            yq = None
            nq = self.dataset_len

        z, y = batch_predict_pairs(
            model=self.model,
            q=qz,
            bq=self.batch_size,
            yq=yq,
            nq=nq,
            verbose=self.verbose,
        )
        yqz, ycz = y
        y = tf.cast(tf.equal(yqz, tf.transpose(ycz)), tf.int32)

        binary_ranking, index_ranking = rank_labels(y, z, remove_top1=self.remove_top1)

        for i, metric_fn in enumerate(self.metric_funcs):
            metric_name = "{}".format(metric_fn.__name__)
            metric_score = metric_fn(binary_ranking)
            logs[metric_name] = metric_score
