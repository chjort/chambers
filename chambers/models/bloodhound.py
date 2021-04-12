import math

import tensorflow as tf

from chambers.layers.attention import MultiHeadAttention
from chambers.layers.distance import CosineSimilarity
from chambers.layers.normalization import L2Normalization
from chambers.layers.reduce import Sum
from chambers.layers.transformer import Decoder, DecoderLayer
from chambers.models import backbones
from chambers.models.backbones.vision_transformer import _obtain_inputs
from chambers.models.base import PredictDataModel
from chambers.utils.generic import ProgressBar


def _pool(x, method=None, prefix=""):
    if method == "avg":
        x = tf.keras.layers.Cropping1D((1, 0), name=prefix + "sequence_embeddings")(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name=prefix + "avg_pool")(x)
    elif method == "max":
        x = tf.keras.layers.Cropping1D((1, 0), name=prefix + "sequence_embeddings")(x)
        x = tf.keras.layers.GlobalMaxPooling1D(name=prefix + "max_pool")(x)
    elif method == "sum":
        x = tf.keras.layers.Cropping1D((1, 0), name=prefix + "sequence_embeddings")(x)
        x = Sum(axis=1, name=prefix + "sum_pool")(x)
    elif method == "cls":
        x = tf.keras.Sequential(
            [
                tf.keras.layers.Cropping1D((0, x.shape[1] - 1)),
                tf.keras.layers.Reshape([-1]),
            ],
            name=prefix + "cls_embedding",
        )(x)

    return x


class _Pool3DAxis1(tf.keras.layers.Layer):
    def __init__(self, method=None, keepdims=False, prefix=None):
        name = method + "_pool" if method is not None else "identity"
        name = prefix + name if prefix is not None else name

        super(_Pool3DAxis1, self).__init__(name=name)
        self.method = method
        self.keepdims = keepdims
        self.axis = 1

    def call(self, inputs, **kwargs):
        if self.method is None:
            return inputs

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


class Matmul(tf.keras.layers.Layer):
    def compute_output_shape(self, input_shape):
        shape_a, shape_b = input_shape
        return [shape_a[0], shape_b[0]]

    def call(self, inputs, **kwargs):
        a, b = inputs
        return tf.matmul(a, b, transpose_b=True)


def Bloodhound0(
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
    feature_dim=None,
    model_name=None,
):
    """
    q -> encoder -> -----> zq
                  |
                  v
          c -> decoder -> zc
    """
    enc = backbones.ViTB16(
        weights="imagenet21k+_224",
        include_top=False,
    )
    patch_size = enc.get_layer("patch_embeddings").get_layer("embedding").kernel_size

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

    enc = tf.keras.Model(
        inputs=enc.inputs, outputs=enc.get_layer("encoder").output, name="encoder"
    )
    embed = tf.keras.Model(
        inputs=enc.inputs, outputs=enc.get_layer("pos_embedding").output, name="embed"
    )
    q_enc = enc(inputs_q)
    c_enc = embed(inputs_c)

    x_c = tf.keras.layers.Dropout(dropout_rate)(c_enc)
    x_c = Decoder(
        embed_dim=x_c.shape[-1],
        num_heads=n_heads,
        ff_dim=ff_dim,
        num_layers=n_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        norm_output=True,
        pre_norm=False,
        causal=False,
    )([x_c, q_enc])

    x_q = _Pool3DAxis1(method=pooling, prefix="q_")(q_enc)
    x_c = _Pool3DAxis1(method=pooling, prefix="c_")(x_c)

    if include_top:
        if pooling is None:
            raise ValueError(
                "`include_top=True` requires `pooling` to be either 'avg', 'max', 'sum', or 'cls'."
            )

        if feature_dim is not None:
            x_q = tf.keras.layers.Dense(feature_dim)(x_q)
            x_c = tf.keras.layers.Dense(feature_dim)(x_c)

        x_q = L2Normalization(axis=-1)(x_q)
        x_c = L2Normalization(axis=-1)(x_c)
        x = Matmul()([x_q, x_c])
    else:
        x = [x_q, x_c]

    model = tf.keras.Model(inputs=[inputs_q, inputs_c], outputs=x, name=model_name)
    return model


def Bloodhound1(
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
    feature_dim=None,
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
        include_top=False,
    )
    patch_size = enc.get_layer("patch_embeddings").get_layer("embedding").kernel_size

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

    enc = tf.keras.Model(
        inputs=enc.inputs, outputs=enc.get_layer("encoder").output, name="encoder"
    )
    q_enc = enc(inputs_q)
    c_enc = enc(inputs_c)
    # TODO: positional embedding here too?
    x_c = Decoder(
        embed_dim=c_enc.shape[-1],
        num_heads=n_heads,
        ff_dim=ff_dim,
        num_layers=n_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        norm_output=True,
        pre_norm=False,
        causal=False,
    )([c_enc, q_enc])

    x_q = _Pool3DAxis1(method=pooling, prefix="q_")(q_enc)
    x_c = _Pool3DAxis1(method=pooling, prefix="c_")(x_c)

    if include_top:
        if pooling is None:
            raise ValueError(
                "`include_top=True` requires `pooling` to be either 'avg', 'max', 'sum', or 'cls'."
            )

        if feature_dim is not None:
            x_q = tf.keras.layers.Dense(feature_dim)(x_q)
            x_c = tf.keras.layers.Dense(feature_dim)(x_c)

        x_q = L2Normalization(axis=-1)(x_q)
        x_c = L2Normalization(axis=-1)(x_c)
        x = Matmul()([x_q, x_c])
    else:
        x = [x_q, x_c]

    model = tf.keras.Model(inputs=[inputs_q, inputs_c], outputs=x, name=model_name)
    return model


#%%


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
    feature_dim=None,
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
        include_top=False,
    )
    patch_size = enc.get_layer("patch_embeddings").get_layer("embedding").kernel_size

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

    enc = tf.keras.Model(
        inputs=enc.inputs, outputs=enc.get_layer("encoder").output, name="encoder"
    )
    q_enc = enc(inputs_q)
    c_enc = enc(inputs_c)
    # NOTE: positional embedding here if encoder is not a transformer
    q_enc = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1), name="q_expand")(
        q_enc
    )
    c_enc = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 0), name="c_expand")(
        c_enc
    )
    x_c = Decoder4D(
        embed_dim=c_enc.shape[-1],
        num_heads=n_heads,
        ff_dim=ff_dim,
        num_layers=n_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        norm_output=True,
        pre_norm=False,
        causal=False,
    )([c_enc, q_enc])

    x_q = _Pool4DAxis2(method=pooling, prefix="q_")(q_enc)
    x_c = _Pool4DAxis2(method=pooling, prefix="c_")(x_c)

    if include_top:
        if pooling is None:
            raise ValueError(
                "`include_top=True` requires `pooling` to be either 'avg', 'max', 'sum', or 'cls'."
            )

        if feature_dim is not None:
            x_q = tf.keras.layers.Dense(feature_dim)(x_q)
            x_c = tf.keras.layers.Dense(feature_dim)(x_c)

        x = CosineSimilarity(axis=-1)([x_q, x_c])
    else:
        x = [x_q, x_c]

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
        n = len(x)
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


def batch_predict(model, q, c, bq, bc=None, yq=None, yc=None, nq=None, nc=None, verbose=True):
    model = PredictDataModel.from_model(model)

    if (yq is None and yc is not None) or (yc is None and yq is not None):
        raise ValueError("Must have both `yq` and `yc` or have both be None.")

    qd, nq = _to_dataset(q, yq, nq)
    cd, nc = _to_dataset(c, yc, nc)
    with_labels = not isinstance(qd.element_spec, tf.TensorSpec)

    if bc is None:
        bc = bq
    qd = qd.batch(bq)
    cd = cd.batch(bc)

    nqb = math.ceil(nq / bq)
    ncb = math.ceil(nc / bc)

    if with_labels:
        repeat_batch = lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(ncb)
    else:
        repeat_batch = lambda x: tf.data.Dataset.from_tensors(x).repeat(ncb)

    qd = qd.flat_map(repeat_batch)
    cd = cd.repeat(nqb)

    if with_labels:
        td = tf.data.Dataset.zip((qd, cd))
        td = td.map(lambda q, c: ((q[0], c[0]), (q[1], c[1])))
    else:
        td = tf.data.Dataset.zip(((qd, cd),))

    if verbose:
        prog = ProgressBar(total=nqb * ncb)
        td = td.apply(prog.dataset_apply_fn)

    if with_labels:
        z, (yqz, ycz) = model.predict(td)
    else:
        z = model.predict(td)

    # predict
    z = tf.reshape(z, [nqb, ncb, bq, bc])
    z = tf.transpose(z, [0, 2, 1, 3])  # [nqb, bq, ncb, bc]
    z = tf.reshape(z, [nq, nc])

    if with_labels:
        yq = tf.reshape(yqz, [nqb, ncb, bq])[:, 0]
        yq = tf.reshape(yq, [-1, 1])
        yc = ycz[:nc]

        yz = tf.cast(tf.equal(yq, tf.transpose(yc)), tf.int32)

        return z, yz

    return z


def batch_predict2(model, q, c, bq, bc, yq=None, yc=None, nq=None, nc=None):
    model = PredictDataModel.from_model(model)

    if (yq is None and yc is not None) or (yc is None and yq is not None):
        raise ValueError("Must have both `yq` and `yc` or have both be None.")

    qd, nq = _to_dataset(q, yq, nq)
    cd, nc = _to_dataset(c, yc, nc)
    with_labels = not isinstance(qd.element_spec, tf.TensorSpec)

    if bc is None:
        bc = bq
    qd = qd.batch(bq)
    cd = cd.batch(bc)

    nqb = math.ceil(nq / bq)
    ncb = math.ceil(nc / bc)

    z = []
    yq = []
    for batch in qd:
        if with_labels:
            qi, yqi = batch
            yq.append(yqi)
        else:
            qi = batch

        qi = tf.data.Dataset.from_tensors(qi).repeat(ncb)

        if with_labels:
            qi_c = tf.data.Dataset.zip((qi, cd)).map(lambda q, c: ((q, c[0]), c[1]))
            zi, yc = model.predict(qi_c)
        else:
            qi_c = tf.data.Dataset.zip(((qi, cd),))
            zi = model.predict(qi_c)

        zi = tf.concat(tf.split(zi, nqb), axis=-1)  # TODO: Error when bq=2, bc=5, nq=10, nc=10
        z.append(zi)
    z = tf.concat(z, axis=0)

    if with_labels:
        yq = tf.concat(yq, axis=0)
        yq = tf.expand_dims(yq, 1)
        yz = tf.cast(tf.equal(yq, tf.transpose(yc)), tf.int32)
        return z, yz

    return z
