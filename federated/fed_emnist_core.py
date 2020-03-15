import collections

import tensorflow as tf
import tensorflow_federated as tff

tff.federated_mean()

N_CLIENTS = 10
EPOCHS_PER_ROUND = 5
N_ROUNDS = 20
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH = 10


def preprocess(dataset):
    td = dataset.repeat(EPOCHS_PER_ROUND).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

    def _format(x):
        x = (tf.reshape(x["pixels"], (-1, 784)),
             tf.one_hot(x["label"], depth=10, dtype=tf.int32))
        return x

    td = td.map(_format)
    return td.prefetch(PREFETCH)


def make_federated_data(client_data, client_ids):
    fed_data = []
    for id_ in client_ids:
        client_dataset = client_data.create_tf_dataset_for_client(id_)
        # client_dataset = preprocess(client_dataset)
        fed_data.append(client_dataset)
    return fed_data


train_data, test_data = tff.simulation.datasets.emnist.load_data()
client_ids = train_data.client_ids[:N_CLIENTS]

train_data = train_data.preprocess(preprocess)
fed_train_set = make_federated_data(train_data, client_ids)

test_data = test_data.preprocess(preprocess)
test_set = test_data.create_tf_dataset_from_all_clients()

# %%

FederatedVariables = collections.namedtuple('FederatedVariables', 'weights bias num_examples loss_sum accuracy_sum')


def create_federated_variables():
    fed_variables = FederatedVariables(
        weights=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
            name='weights',
            trainable=True),
        bias=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(10)),
            name='bias',
            trainable=True),
        num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
        loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
        accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))
    return fed_variables


def get_local_mnist_metrics(variables):
    return collections.OrderedDict(
        num_examples=variables.num_examples,
        loss=variables.loss_sum / variables.num_examples,
        accuracy=variables.accuracy_sum / variables.num_examples)


@tff.federated_computation
def aggregate_mnist_metrics_across_clients(metrics):
    return collections.OrderedDict(
        num_examples=tff.federated_sum(metrics.num_examples),
        loss=tff.federated_mean(metrics.loss, metrics.num_examples),
        accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples))


# %% forward pass
def mnist_forward_pass(variables, batch):
    y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
    predictions = tf.cast(tf.argmax(y, 1), tf.int32)

    flat_labels = tf.reshape(batch['y'], [-1])
    loss = -tf.reduce_mean(
        tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, flat_labels), tf.float32))

    num_examples = tf.cast(tf.size(batch['y']), tf.float32)

    variables.num_examples.assign_add(num_examples)
    variables.loss_sum.assign_add(loss * num_examples)
    variables.accuracy_sum.assign_add(accuracy * num_examples)

    return loss, predictions


class MnistModel(tff.learning.Model):

    def __init__(self):
        self._variables = create_federated_variables()

    @property
    def trainable_variables(self):
        return [self._variables.weights, self._variables.bias]

    @property
    def non_trainable_variables(self):
        return []

    @property
    def local_variables(self):
        return [
            self._variables.num_examples, self._variables.loss_sum,
            self._variables.accuracy_sum
        ]

    @property
    def input_spec(self):
        return collections.OrderedDict(
            x=tf.TensorSpec([None, 784], tf.float32),
            y=tf.TensorSpec([None, 1], tf.int32))

    @tf.function
    def forward_pass(self, batch, training=True):
        del training
        loss, predictions = mnist_forward_pass(self._variables, batch)
        num_exmaples = tf.shape(batch['x'])[0]
        return tff.learning.BatchOutput(
            loss=loss, predictions=predictions, num_examples=num_exmaples)

    @tf.function
    def report_local_outputs(self):
        return get_local_mnist_metrics(self._variables)

    @property
    def federated_output_computation(self):
        return aggregate_mnist_metrics_across_clients
