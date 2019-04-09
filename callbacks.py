import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from chambers import metrics, metrics_np


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


class LoggerNumpy(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, config_dir):
        super().__init__()
        self.log_dir = log_dir
        self.config_dir = config_dir
        self.log_cache = None
        self.logfile = None
        self.start_time = None
        self.sess = None
        self.values = ["loss"]

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.logfile = os.path.join(self.log_dir, "logfile.txt")
        self.sess = tf.keras.backend.get_session()
        tf.train.write_graph(self.sess.graph_def, self.config_dir, "graph.pbtxt")

    def on_epoch_end(self, epoch, logs=None):
        # write logs
        if epoch == 0:
            if "val_loss" in logs:
                self.values.append("val_loss")
            for metric in metrics_np.available_metrics():
                if metric in logs:
                    self.values.append(metric)
            self.log_cache = {v: [] for v in self.values}
            self.log_cache["epoch"] = []
            with open(self.logfile, "w") as f:
                f.write("epoch," + ",".join(self.values) + "\n")

        self.log_cache["epoch"].append(epoch)
        with open(self.logfile, "a") as f:
            f.write(str(epoch))
            for v in self.values:
                self.log_cache[v].append(logs[v])
                f.write("," + str(logs[v]))
            f.write("\n")

        # plot figures
        plt.figure()
        plt.plot(self.log_cache["loss"], label="Training loss")
        if "val_loss" in self.log_cache:
            plt.plot(self.log_cache["val_loss"], label="Validation loss")
        plt.title("Loss [{}]".format(self.model.loss))
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, "losses.png"), transparent=True)

        plt.clf()
        if "acc_np" in self.log_cache:
            plt.plot(self.log_cache["acc_np"], label="Global accuracy")
        if "class_acc_np" in self.log_cache:
            for i in range(np.shape(self.log_cache["class_acc_np"])[1]):
                class_acc = [x[i] for x in self.log_cache["class_acc_np"]]
                plt.plot(class_acc, label="class {} accuracy".format(i))
        if "acc_np" in self.log_cache or "class_acc_np" in self.log_cache:
            plt.title("Validation accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.legend()
            plt.savefig(os.path.join(self.log_dir, "accuracies.png"), transparent=True)

        plt.clf()
        for value in self.values:
            if value not in {"loss", "val_loss", "acc_np", "class_acc_np"}:
                plt.plot(self.log_cache[value])
                plt.title("Validation {}".format(value))
                plt.ylabel(value)
                plt.xlabel("Epoch")
                plt.savefig(os.path.join(self.log_dir, "{}.png".format(value)), transparent=True)
                plt.clf()
        plt.close("all")

    def on_train_end(self, logs=None):
        training_time = time.time() - self.start_time
        with open(self.logfile, "a") as f:
            f.write("Total training time: " + str(training_time) + "\n")


class MetricsNumpy(tf.keras.callbacks.Callback):
    def __init__(self, metric_list):
        super().__init__()
        self.metrics = metric_list
        self.sess = None

    def on_train_begin(self, logs=None):
        self.sess = tf.keras.backend.get_session()

    def on_epoch_end(self, epoch, logs=None):
        metrics_scores = {metric: [] for metric in self.metrics}
        for i in range(self.params["validation_steps"]):
            x, y_true = self.sess.run([self.validation_data[0],  # val x batch
                                       self.validation_data[1]])  # val y batch
            y_pred = self.model.predict(x)
            for metric in self.metrics:
                metric_func = metrics_np.get_metric_np(metric)
                score = metric_func(y_true, y_pred)
                metrics_scores[metric].append(score)

        for metric in self.metrics:
            mean_score = np.mean(metrics_scores[metric], axis=0).tolist()
            logs[metric] = mean_score