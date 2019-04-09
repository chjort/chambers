import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time


class ValProgbar(tf.keras.callbacks.ProgbarLogger):
    """
        Progress bar with training metrics omitted.
    """

    def __init__(self, verbose, count_mode='steps'):
        self.verbose = verbose
        super(ValProgbar, self).__init__(count_mode)

    def on_train_begin(self, logs=None):
        # filter out the training metrics
        self.params['metrics'] = [m for m in self.params['metrics'] if (m == "loss" or m.startswith('val_'))]
        self.epochs = self.params['epochs']


class Logger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, config_dir, plot=True):
        super().__init__()
        self.config_dir = config_dir
        self.log_dir = log_dir
        self.log_cache = {}
        self.logfile = None
        self.start_time = None
        self.fields = None
        self.sess = None
        self.make_plots = plot

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.logfile = open(os.path.join(self.log_dir, "logfile.txt"), "w")
        self.log_cache["epoch"] = []
        self.log_cache = {v: [] for v in self.params["metrics"]}
        self.logfile.write("epoch," + ",".join(self.params["metrics"]) + "\n")
        self.sess = tf.keras.backend.get_session()
        tf.train.write_graph(self.sess.graph_def, self.config_dir, "graph.pbtxt")

    def on_epoch_end(self, epoch, logs=None):
        self.logfile.write(str(epoch))
        for metric in self.params["metrics"]:
            self.logfile.write("," + str(logs[metric]))
            self.log_cache[metric].append(logs[metric])
        self.logfile.write("\n")
        self.logfile.flush()

        if self.make_plots:
            fig = plt.figure()
            if "val_loss" in self.log_cache:
                self.plot(series=[self.log_cache["loss"], self.log_cache["val_loss"]],
                          labels=["Training loss", "Validation loss"],
                          title="loss_{}".format(self.model.loss.__name__),
                          ylabel="Loss"
                          )
            else:
                self.plot(series=[self.log_cache["loss"]], labels=["Training loss"],
                          title="loss_{}".format(self.model.loss.__name__),
                          ylabel="Loss"
                          )
            plt.clf()
            for metric in self.params["metrics"]:
                if metric != "loss" and metric != "val_loss":
                    self.plot(series=[self.log_cache[metric]], labels=[metric],
                              title=metric,
                              ylabel=metric
                              )
                plt.clf()
            plt.close("all")

    def on_train_end(self, logs=None):
        training_time = time.time() - self.start_time
        self.logfile.write("Total training time: " + str(training_time) + "\n")
        self.logfile.close()

    def plot(self, series, labels, title=None, ylabel=None):
        assert len(series) == len(labels)
        for serie, label in zip(series,labels):
            plt.plot(serie, label=label)
        if title is not None:
            plt.title(title)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, title+".png"), transparent=True)
