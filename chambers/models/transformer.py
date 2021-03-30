import tensorflow as tf
from einops.layers.tensorflow import Rearrange

from chambers.activations import gelu
from chambers.layers.embedding import (
    PositionalEmbedding1D,
    ConcatEmbedding,
    LearnedEmbedding1D,
    LearnedEmbedding0D,
)
from chambers.layers.transformer import Encoder, Decoder


def Seq2SeqTransformer(
    input_vocab_size,
    output_vocab_size,
    embed_dim,
    num_heads,
    dim_feedforward,
    num_encoder_layers,
    num_decoder_layers,
    dropout_rate=0.1,
    name="seq2seq_transformer",
):
    inputs = tf.keras.layers.Input(shape=(None,), name="inputs_tokens")
    targets = tf.keras.layers.Input(shape=(None,), name="targets_tokens")

    x_enc = tf.keras.layers.Embedding(
        input_vocab_size, embed_dim, mask_zero=True, name="inputs_embed"
    )(inputs)
    x_enc = PositionalEmbedding1D(embed_dim, name="inputs_positional_encoding")(x_enc)
    x_enc = Encoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=dim_feedforward,
        num_layers=num_encoder_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
    )(x_enc)

    x_dec = tf.keras.layers.Embedding(
        output_vocab_size, embed_dim, mask_zero=True, name="targets_embed"
    )(targets)
    x_dec = PositionalEmbedding1D(embed_dim, name="targets_positional_encoding")(x_dec)
    x_dec = Decoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=dim_feedforward,
        num_layers=num_decoder_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        norm_output=False,
        causal=True,
    )([x_dec, x_enc])

    x = tf.keras.layers.Dense(output_vocab_size)(x_dec)

    model = tf.keras.models.Model(inputs=[inputs, targets], outputs=x, name=name)
    return model


def _patch_embeddings(x, patch_size, patch_dim, name=None):
    x = tf.keras.Sequential(
        [
            Rearrange(
                "b (h p1) (w p2) c -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            tf.keras.layers.Dense(patch_dim),
        ],
        name=name,
    )(x)
    # x = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.Conv2D(
    #             filters=patch_dim,
    #             kernel_size=patch_size,
    #             strides=patch_size,
    #             padding="valid",
    #         ),
    #         tf.keras.layers.Reshape([-1, patch_dim])
    #     ],
    #     name=name,
    # )(x)
    return x


def VisionTransformer(
    input_shape,
    n_classes,
    patch_size,
    patch_dim,
    n_encoder_layers,
    n_heads,
    ff_dim,
    dropout_rate=0.0,
):
    inputs = tf.keras.layers.Input(input_shape)
    x = _patch_embeddings(inputs, patch_size, patch_dim, name="patch_embeddings")
    x = ConcatEmbedding(
        n_embeddings=1,
        embedding_dim=patch_dim,
        side="left",
        axis=1,
        initializer=tf.keras.initializers.RandomNormal(),
        name="add_cls_token",
    )(x)
    x = LearnedEmbedding1D(
        initializer=tf.keras.initializers.RandomNormal(), name="pos_embedding"
    )(x)
    x = Encoder(
        embed_dim=patch_dim,
        num_heads=n_heads,
        ff_dim=ff_dim,
        num_layers=n_encoder_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        norm_output=True,
    )(x)
    x = tf.keras.layers.Cropping1D((0, x.shape[1] - 1))(x)
    x = tf.keras.layers.Reshape([-1])(x)

    x = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(ff_dim, activation=gelu),
            tf.keras.layers.Dense(n_classes),
        ],
        name="mlp_head",
    )(x)

    model = tf.keras.models.Model(inputs, x)
    return model


tf.keras.utils.get_custom_objects().update(
    {
        "Rearrange": Rearrange,
    }
)
