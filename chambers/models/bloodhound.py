import tensorflow as tf
from tensorflow.python.keras.utils import layer_utils

from chambers.layers.attention import MultiHeadAttention
from chambers.layers.distance import CosineSimilarity
from chambers.layers.transformer import Decoder, DecoderLayer
from chambers.models import backbones
from chambers.models.backbones.vision_transformer import _obtain_inputs
from chambers.utils.generic import ProgressBar


@tf.keras.utils.register_keras_serializable(package="Chambers")
class _Pool3DAxis1(tf.keras.layers.Layer):
    def __init__(self, method="avg", keepdims=False, name=None, **kwargs):
        super(_Pool3DAxis1, self).__init__(name=name, **kwargs)
        if method not in {"avg", "max", "sum", "cls"}:
            raise ValueError("`method` must be either 'avg', 'max', 'sum' or 'cls'.")

        self.method = method
        self.keepdims = keepdims
        self.axis = 1

    def call(self, inputs, **kwargs):
        x = self._slice(inputs)

        if self.method == "avg":
            x = tf.reduce_mean(x, axis=self.axis, keepdims=self.keepdims)
        elif self.method == "max":
            x = tf.reduce_max(x, axis=self.axis, keepdims=self.keepdims)
        elif self.method == "sum":
            x = tf.reduce_sum(x, axis=self.axis, keepdims=self.keepdims)

        return x

    def _slice(self, x):
        if self.method == "cls":
            x = x[:, 0, :]
        else:
            x = x[:, 1:, :]
        return x

    def get_config(self):
        config = {
            "method": self.method,
            "keepdims": self.keepdims,
        }
        base_config = super(_Pool3DAxis1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="Chambers")
class _Pool4DAxis2(_Pool3DAxis1):
    def __init__(self, *args, **kwargs):
        super(_Pool4DAxis2, self).__init__(*args, **kwargs)
        self.axis = 2

    def _slice(self, x):
        if self.method == "cls":
            x = x[:, :, 0, :]
        else:
            x = x[:, :, 1:, :]
        return x


@tf.keras.utils.register_keras_serializable(package="Chambers")
class ExpandDims(tf.keras.layers.Layer):
    def __init__(self, axis=0, **kwargs):
        super(ExpandDims, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.expand_dims(inputs, self.axis)

    def get_config(self):
        config = {
            "axis": self.axis,
        }
        base_config = super(ExpandDims, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="Chambers")
class MultiHeadAttention4D(MultiHeadAttention):
    def call(self, inputs, mask=None, training=None):
        """
        Einsum notation:
        b = batch_size
        e = expanded dimension
        t = sequence length
        d = embedding dimension
        n = num heads
        h = head dimension
        """
        q = inputs[0]  # [b, e, tq, d]
        v = inputs[1]  # [e, b, tv, d]
        k = inputs[2] if len(inputs) > 2 else v  # [e, b, tv, d]

        # linear projections + head split
        query = tf.einsum("ebtd,dnh->ebnth", q, self.w_query) + self.b_query
        value = tf.einsum("betd,dnh->benth", v, self.w_value) + self.b_value
        key = tf.einsum("betd,dnh->benth", k, self.w_key) + self.b_key

        # TODO: Mask

        # TODO: Make attention attend across query batch (b) dimension for each candidate
        attention = self.attention([query, value, key], mask=mask, training=training)

        # linear projection + head merge
        x = (
            tf.einsum("benth,ndh->betd", attention, self.w_projection)
            + self.b_projection
        )

        return x


@tf.keras.utils.register_keras_serializable(package="Chambers")
class DecoderLayer4D(DecoderLayer):
    def __init__(self, *args, **kwargs):
        super(DecoderLayer4D, self).__init__(*args, **kwargs)

        # self-attention
        self.multi_head_attention1 = MultiHeadAttention4D(
            head_dim=self.embed_dim // self.num_heads,
            num_heads=self.num_heads,
            dense_kernel_initializer=self.dense_kernel_initializer,
            dropout_rate=self.attention_dropout_rate,
            causal=self.causal,
        )

        # cross-attention
        self.multi_head_attention2 = MultiHeadAttention4D(
            head_dim=self.embed_dim // self.num_heads,
            num_heads=self.num_heads,
            dense_kernel_initializer=self.dense_kernel_initializer,
            dropout_rate=self.attention_dropout_rate,
            causal=False,
        )


@tf.keras.utils.register_keras_serializable(package="Chambers")
class Decoder4D(Decoder):
    def build(self, input_shape):
        self.layers = [
            DecoderLayer4D(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dense_kernel_initializer=self.dense_kernel_initializer,
                attention_dropout_rate=self.attention_dropout_rate,
                dense_dropout_rate=self.dense_dropout_rate,
                norm_epsilon=self.norm_epsilon,
                pre_norm=self.pre_norm,
                causal=self.causal,
            )
            for i in range(self.num_layers)
        ]


def BloodhoundFunctional(
    query_shape,
    candidates_shape,
    n_layers,
    n_heads,
    ff_dim,
    dropout_rate=0.1,
    include_top=True,
    pooling=None,
    name=None,
):
    """
        q -> encoder -> -----> zq
                      |
                      v
    c -> encoder -> decoder -> zc
    """

    inputs_q = tf.keras.layers.Input(shape=query_shape)
    inputs_c = tf.keras.layers.Input(shape=candidates_shape)

    # NOTE: positional embedding here if encoder is not a transformer
    q = ExpandDims(axis=1, name="q_expand")(inputs_q)
    c = ExpandDims(axis=0, name="c_expand")(inputs_c)
    c = Decoder4D(
        embed_dim=c.shape[-1],
        num_heads=n_heads,
        ff_dim=ff_dim,
        num_layers=n_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        norm_output=True,
        pre_norm=False,
        causal=False,
        name="decoder",
    )([c, q])

    if pooling is not None:
        q = _Pool4DAxis2(method=pooling, name="pool_q")(q)
        c = _Pool4DAxis2(method=pooling, name="pool_c")(c)

    if include_top:
        if pooling is None:
            raise ValueError(
                "`include_top=True` requires `pooling` to be either 'avg', 'max', 'sum', or 'cls'."
            )

        x = CosineSimilarity(axis=-1)([q, c])
        # x = tf.keras.layers.Dense(1, activation="sigmoid", dtype=tf.float32)(c)
        # x = tf.keras.layers.Lambda(lambda x: x[:, :, 0])(x)
    else:
        x = [q, c]

    model = tf.keras.Model(inputs=[inputs_q, inputs_c], outputs=x, name=name)
    return model


def Bloodhound4D(
    n_layers,
    n_heads,
    ff_dim,
    dropout_rate=0.1,
    query_tensor=None,
    query_shape=None,
    candidates_tensor=None,
    candidates_shape=None,
    include_top=True,
    weights="imagenet21k+_224",
    pooling=None,
    model_name=None,
):
    """
        q -> encoder -> -----> zq
                      |
                      v
    c -> encoder -> decoder -> zc
    """
    enc = backbones.ViTB16(
        weights="imagenet21k+_224",
        pooling=None,
        include_top=False,
    )
    patch_size = enc.get_layer("patch_embeddings").get_layer("embedding").kernel_size
    enc = tf.keras.Model(enc.inputs, enc.outputs, name="encoder")

    inputs_q = _obtain_inputs(
        query_tensor,
        query_shape,
        default_size=224,
        min_size=patch_size,
        weights=weights,
        model_name=model_name,
        name="query",
    )
    inputs_c = _obtain_inputs(
        candidates_tensor,
        candidates_shape,
        default_size=224,
        min_size=patch_size,
        weights=weights,
        model_name=model_name,
        name="candidates",
    )

    q_enc = enc(inputs_q)
    c_enc = enc(inputs_c)
    # NOTE: positional embedding here if encoder is not a transformer
    dec = BloodhoundFunctional(
        query_shape=q_enc.shape[1:],
        candidates_shape=c_enc.shape[1:],
        n_layers=n_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate,
        include_top=include_top,
        pooling=pooling,
        name="decoder",
    )
    x = dec([q_enc, c_enc])

    x = tf.keras.layers.Activation("linear", dtype=tf.float32, name="cast_float32")(x)

    if query_tensor is not None:
        inputs_q = layer_utils.get_source_inputs(query_tensor)
    if candidates_tensor is not None:
        inputs_c = layer_utils.get_source_inputs(candidates_tensor)

    model = tf.keras.Model(inputs=[inputs_q, inputs_c], outputs=x, name=model_name)
    return model

#%%
def valid_cardinality(dataset):
    if dataset.cardinality() == tf.data.INFINITE_CARDINALITY:
        return False
    if dataset.cardinality() == tf.data.UNKNOWN_CARDINALITY:
        return False
    return True


def _to_dataset(x, y=None, n=None):
    if not isinstance(x, tf.data.Dataset):
        n = tf.shape(x)[0]
        if y is not None:
            x = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            x = tf.data.Dataset.from_tensor_slices(x)
    else:
        if valid_cardinality(x):
            n = x.cardinality()
        elif not valid_cardinality(x) and n is None:
            raise ValueError("Unable to infer length of dataset {}.".format(x))

    return x, n


def pair_iteration_dataset(q, c, bq, bc, yq=None, yc=None, nq=None, nc=None):
    qd, nq = _to_dataset(q, yq, nq)
    cd, nc = _to_dataset(c, yc, nc)
    with_labels = not isinstance(qd.element_spec, tf.TensorSpec)

    bq = tf.cast(bq, tf.int64)
    bc = tf.cast(bc, tf.int64)
    nq = tf.cast(nq, tf.int64) if nq is not None else nq
    nc = tf.cast(nc, tf.int64) if nc is not None else nc

    qd = qd.batch(bq)
    cd = cd.batch(bc)

    nqb = tf.cast(tf.math.ceil(nq / bq), tf.int64)
    ncb = tf.cast(tf.math.ceil(nc / bc), tf.int64)

    if with_labels:
        repeat_batch = lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(ncb)
    else:
        repeat_batch = lambda x: tf.data.Dataset.from_tensors(x).repeat(ncb)

    qd = qd.flat_map(repeat_batch)
    cd = cd.repeat(nqb)

    if with_labels:
        td = tf.data.Dataset.zip((qd, cd))
        # ((x_q, x_c), (y_q, y_c))
        td = td.map(lambda q, c: ((q[0], c[0]), (q[1], c[1])))
    else:
        td = tf.data.Dataset.zip(((qd, cd),))

    return td


def reshape_pair_predictions(x, bq, bc, nq, nc, y=None):
    nqb = tf.cast(tf.math.ceil(nq / bq), tf.int64)
    ncb = tf.cast(tf.math.ceil(nc / bc), tf.int64)

    x = tf.reshape(x, [nqb, ncb, bq, bc])
    x = tf.transpose(x, [0, 2, 1, 3])  # [nqb, bq, ncb, bc]
    x = tf.reshape(x, [nq, nc])

    if y is not None:
        yq, yc = y
        yq = tf.reshape(yq, [nqb, ncb, bq])[:, 0]
        yq = tf.reshape(yq, [-1, 1])
        yc = yc[:nc]
        return x, (yq, yc)

    return x


def batch_predict_pairs(
    model, q, bq, c=None, bc=None, yq=None, yc=None, nq=None, nc=None, verbose=True
):
    if c is None:
        c = q
        bc = bq
        yc = yq
        nc = nq
    elif bc is None:
        bc = bq

    q, nq = _to_dataset(q, yq, nq)
    c, nc = _to_dataset(c, yc, nc)

    bq = tf.cast(bq, tf.int64)
    bc = tf.cast(bc, tf.int64)
    nq = tf.cast(nq, tf.int64) if nq is not None else nq
    nc = tf.cast(nc, tf.int64) if nc is not None else nc

    bq = tf.minimum(bq, nq)
    bc = tf.minimum(bc, nc)

    td = pair_iteration_dataset(q, c, bq, bc, yq, yc, nq, nc)

    if verbose:
        nqb = tf.cast(tf.math.ceil(nq / bq), tf.int32)
        ncb = tf.cast(tf.math.ceil(nc / bc), tf.int32)

        prog = ProgressBar(total=nqb * ncb)
        td = td.apply(prog.dataset_apply_fn)

    z = model.predict(td)

    if isinstance(z, tuple):
        z, y = z
        z, y = reshape_pair_predictions(z, bq, bc, nq, nc, y=y)
        return z, y

    z = reshape_pair_predictions(z, bq, bc, nq, nc)
    return z
