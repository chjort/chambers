import tensorflow as tf
import tensorflow_federated as tff

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
def MLP():
    inputs = tf.keras.layers.Input(shape=(784,))
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

    # TODO: Remove the sample batch in tensorflow_federated==0.13.0
    sample_batch = (tf.zeros(shape=(BATCH_SIZE, 784), dtype=tf.float32),
                    tf.zeros(shape=(BATCH_SIZE, 10), dtype=tf.int32))
    tff_model = tff.learning.from_keras_model(keras_model=model,
                                              dummy_batch=sample_batch,
                                              loss=tf.keras.losses.CategoricalCrossentropy(),
                                              metrics=[tf.keras.metrics.CategoricalAccuracy()]
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


# %%
training_process = tff.learning.build_federated_averaging_process(model_fn=build_model,
                                                                  client_optimizer_fn=build_client_opt,
                                                                  server_optimizer_fn=build_server_opt
                                                                  )
print(training_process.initialize.type_signature.formatted_representation())

state = training_process.initialize()
# state[0][0] # model trainable weights
# state[0][1] # model non_trainable weights
# state[2] # optimizer state (parameters for client optimizers)
# state[3] # delta_aggregate_state
# state[4] # model_broadcast_state

for i in range(N_ROUNDS):
    state, metrics = training_process.next(state, fed_train_set)
    print(metrics)

#%%
model = MLP()
model.compile(optimizer="sgd",
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=["acc"])

model.set_weights(state[0][0])

model.evaluate(test_set, steps=500)