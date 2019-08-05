import tensorflow as tf
import matplotlib.pyplot as plt
import os


class ValProgbar(tf.keras.callbacks.ProgbarLogger):
    """
        Progress bar only showing validation metrics
    """

    def __init__(self, verbose, count_mode='steps'):
        self.verbose = verbose
        super(ValProgbar, self).__init__(count_mode)

    def on_train_begin(self, logs=None):
        # filter out the training metrics
        self.params['metrics'] = [m for m in self.params['metrics'] if (m == "loss" or m.startswith('val_'))]
        self.verbose = self.params['verbose']
        self.epochs = self.params['epochs']


class StopTrainingOnDemand(tf.keras.callbacks.Callback):
    """Callback that terminates training when flag=1 is encountered.
    """

    def __init__(self):
        super().__init__()
        self.flag = 0

    def on_train_begin(self, logs=None):
        self.flag = 0

    def on_batch_end(self, batch, logs=None):
        if self.flag == 1:
            print("\nPausing training.")
            self.model.stop_training = True

    def stop_training(self):
        self.flag = 1


class MetricPlotter(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, transparent=True):
        super(MetricPlotter, self).__init__()
        self.save_dir = save_dir
        self.transparent = transparent
        self.logs_history = {}
        self.epochs = []

    def on_train_begin(self, logs=None):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        fig = plt.figure(1)
        for key in logs.keys():
            if not key.startswith("val_"):
                if key in self.logs_history:
                    self.logs_history[key].append(logs[key])
                else:
                    self.logs_history[key] = [logs[key]]
                plt.plot(self.logs_history[key], self.epochs, label="training")

                if "val_" + key in logs:
                    vkey = "val_" + key
                    if vkey in self.logs_history:
                        self.logs_history[vkey].append(logs[vkey])
                    else:
                        self.logs_history[vkey] = [logs[key]]
                    plt.plot(self.logs_history[vkey], self.epochs, label="validation")

                plt.title(key)
                plt.ylabel(key)
                plt.xlabel("Epochs")
                plt.legend()
                plt.savefig(os.path.join(self.save_dir, "{}.png".format(key)), transparent=self.transparent)
                plt.clf()
