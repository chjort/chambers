import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# TODO: Get rid of Scikit-Learn warning "UndefinedMetricWarning"

def get_metric_np(metric_name):
    funcs = {"acc_np": global_accuracy_np,
             "class_acc_np": class_accuracy_np,
             "prec_np": precision_np,
             "rec_np": recall_np,
             "f1_np": f1_np,
             "iou_np": iou_np,
             }
    return funcs[metric_name]


def available_metrics():
    return ("acc_np", "class_acc_np", "prec_np", "rec_np", "f1_np", "iou_np")


def global_accuracy_np(y_true, y_pred):
    batch_size = np.shape(y_true)[0]
    batch_accuracy = []
    for i in range(batch_size):
        batch_accuracy.append(global_accuracy_single_np(y_true[i], y_pred[i]))

    return np.mean(batch_accuracy)


def class_accuracy_np(y_true, y_pred):
    batch_size = np.shape(y_true)[0]
    batch_class_accuracy = []
    for i in range(batch_size):
        batch_class_accuracy.append(class_accuracy_single_np(y_true[i], y_pred[i]))

    return np.mean(batch_class_accuracy, axis=0)


def precision_np(y_true, y_pred):
    batch_size = np.shape(y_true)[0]
    batch_precision = []
    for i in range(batch_size):
        batch_precision.append(precision_single_np(y_true[i], y_pred[i]))

    return np.mean(batch_precision)


def recall_np(y_true, y_pred):
    batch_size = np.shape(y_true)[0]
    batch_recall = []
    for i in range(batch_size):
        batch_recall.append(recall_single_np(y_true[i], y_pred[i]))

    return np.mean(batch_recall)


def f1_np(y_true, y_pred):
    batch_size = np.shape(y_true)[0]
    batch_f1 = []
    for i in range(batch_size):
        batch_f1.append(f1_single_np(y_true[i], y_pred[i]))

    return np.mean(batch_f1)


def iou_np(y_true, y_pred):
    batch_size = np.shape(y_true)[0]
    batch_iou = []
    for i in range(batch_size):
        batch_iou.append(iou_single_np(y_true[i], y_pred[i]))

    return np.mean(batch_iou)


# Compute the average segmentation accuracy across all classes
def global_accuracy_single_np(y_true, y_pred):
    y_true = y_true.argmax(-1).flatten()
    y_pred = y_pred.argmax(-1).flatten()

    total = len(y_true)
    correct = np.sum(y_pred == y_true)
    return correct / total


# Compute the class-specific segmentation accuracy
def class_accuracy_single_np(y_true, y_pred):
    num_classes = y_true.shape[-1]
    y_true = y_true.argmax(-1).flatten()
    y_pred = y_pred.argmax(-1).flatten()

    total = []
    for val in range(num_classes):
        total.append((y_true == val).sum())

    count = [0.0] * num_classes
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            count[int(y_pred[i])] = count[int(y_pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT,
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def precision_single_np(y_true, y_pred):
    y_true = y_true.argmax(-1).flatten()
    y_pred = y_pred.argmax(-1).flatten()
    return precision_score(y_true, y_pred, average="weighted")


def recall_single_np(y_true, y_pred):
    y_true = y_true.argmax(-1).flatten()
    y_pred = y_pred.argmax(-1).flatten()
    return recall_score(y_true, y_pred, average="weighted")


def f1_single_np(y_true, y_pred):
    y_true = y_true.argmax(-1).flatten()
    y_pred = y_pred.argmax(-1).flatten()
    return f1_score(y_true, y_pred, average="weighted")


def iou_single_np(y_true, y_pred):
    y_true = y_true.argmax(-1).flatten()
    y_pred = y_pred.argmax(-1).flatten()

    unique_labels = np.unique(y_true)
    num_unique_labels = len(unique_labels)

    intersection = np.zeros(num_unique_labels)
    union = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = y_pred == val
        label_i = y_true == val

        intersection[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        union[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    mean_iou = np.mean(intersection / union)
    return mean_iou