import tensorflow as tf

from chambers.models.bloodhound import batch_predict
from chambers.utils.ranking import rank_labels


class PairedRankingMetricCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        dataset: tf.data.Dataset,
        len_dataset,
        metric_funcs,
        batch_size=10,
        remove_top1=False,
        model=None,
        verbose=True,
        name="ranking_metrics",
    ):
        super().__init__()
        self.dataset = dataset
        self.len_dataset = len_dataset
        self.metric_funcs = metric_funcs
        self.batch_size = batch_size
        self.remove_top1 = remove_top1
        self.model = model
        self.verbose = verbose
        self.name = name
        self._supports_tf_logs = True

    def set_model(self, model):
        if self.model is None:
            self.model = model

    def on_epoch_end(self, epoch, logs=None):
        z, y = batch_predict(
            model=self.model,
            q=self.dataset,
            c=self.dataset,
            bq=self.batch_size,
            bc=self.batch_size,
            yq=None,
            yc=None,
            nq=self.len_dataset,
            nc=self.len_dataset,
            verbose=self.verbose,
        )

        binary_ranking, index_ranking = rank_labels(y, z, remove_top1=self.remove_top1)

        for i, metric_fn in enumerate(self.metric_funcs):
            metric_name = "{}".format(metric_fn.__name__)
            metric_score = metric_fn(binary_ranking)
            logs[metric_name] = metric_score
