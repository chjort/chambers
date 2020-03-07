import tensorflow as tf
import tensorflow_federated as tff

# %%
def preprocess(dataset):
    batch_size = 20
    shuffle_buffer = 100
    td = dataset.repeat().shuffle(shuffle_buffer).batch(batch_size)
    td = td.map(lambda x: (tf.reshape(x["pixels"], (batch_size, -1)), x["label"]))
    return td.prefetch(10)


def make_federated_data(client_data, client_ids):
    fed_data = []
    for id_ in client_ids:
        client_dataset = client_data.create_tf_dataset_for_client(id_)
        client_dataset = preprocess(client_dataset)
        fed_data.append(client_dataset)
    return fed_data


train_data, test_data = tff.simulation.datasets.emnist.load_data()

client_ids = train_data.client_ids[:10]
fed_data = make_federated_data(train_data, client_ids)

sample_batch = next(iter(fed_data[0]))
# sample_batch = (tf.zeros(shape=(20, 784), dtype=tf.float32),
#                 tf.zeros(shape=(20,), dtype=tf.float32))

# %%
def MLP():
    inputs = tf.keras.layers.Input(shape=(784))
    x = tf.keras.layers.Dense(1024)(inputs)
    x = tf.keras.layers.Dense(10)(x)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model


def build_model():
    """
    Build the federated model used by clients and server.
    :return: Federated model
    """
    model = MLP()
    tff_model = tff.learning.from_keras_model(keras_model=model,
                                              dummy_batch=sample_batch,
                                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                                              )
    return tff_model


def build_client_opt():
    """
    Build the optimizer used to _compute_ the gradients for the model weights from the local client data.
    :return: Client optimizer
    """
    client_opt = tf.keras.optimizers.SGD(learning_rate=0.02)
    return client_opt


def build_server_opt():
    """
    Build the optimizer used to _apply_ the averaged gradients from the clients to the model weights.
    :return: Server optimizer
    """

    # Learning rate of 1 so the client gradients are not scaled further.
    server_opt = tf.keras.optimizers.SGD(learning_rate=1.)
    return server_opt


training_process = tff.learning.build_federated_averaging_process(model_fn=build_model,
                                                                  client_optimizer_fn=build_client_opt,
                                                                  server_optimizer_fn=build_server_opt
                                                                  )
