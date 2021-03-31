import tensorflow as tf

from chambers.layers.embedding import (
    PositionalEmbedding1D,
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


# def VisionTransformerOS(
#     input_shape,
#     n_classes,
#     patch_size,
#     patch_dim,
#     n_encoder_layers,
#     n_heads,
#     ff_dim,
#     dropout_rate=0.0,
# ):
#     inputs1 = tf.keras.layers.Input(input_shape)
#     inputs2 = tf.keras.layers.Input(input_shape)
#
#     patch_embeddings = tf.keras.Sequential(
#         [
#             tf.keras.layers.Conv2D(
#                 filters=patch_dim,
#                 kernel_size=patch_size,
#                 strides=patch_size,
#                 padding="valid",
#                 name="embedding",
#             ),
#             tf.keras.layers.Reshape([-1, patch_dim]),
#         ],
#         name="patch_embeddings",
#     )
#     x1 = patch_embeddings(inputs1)
#     x2 = patch_embeddings(inputs2)
#
#     x1 = LearnedEmbedding0D(
#         initializer=tf.keras.initializers.RandomNormal(stddev=0.06),
#         name="segment1_embedding",
#     )(x1)
#     x2 = LearnedEmbedding0D(
#         initializer=tf.keras.initializers.RandomNormal(stddev=0.06),
#         name="segment2_embedding",
#     )(x2)
#
#     x = ConcatEmbedding(
#         n_embeddings=1,
#         embedding_dim=patch_dim,
#         side="left",
#         axis=1,
#         initializer="zeros",
#         name="add_cls_token",
#     )(x1)
#     x = ConcatEmbedding(
#         n_embeddings=1,
#         embedding_dim=patch_dim,
#         side="right",
#         axis=1,
#         initializer="zeros",
#         name="add_sep_token",
#     )(x)
#     x = tf.keras.layers.Concatenate(axis=1)([x, x2])
#
#     x = LearnedEmbedding1D(
#         initializer=tf.keras.initializers.RandomNormal(stddev=0.06),
#         name="pos_embedding",
#     )(x)
#     x = Encoder(
#         embed_dim=patch_dim,
#         num_heads=n_heads,
#         ff_dim=ff_dim,
#         num_layers=n_encoder_layers,
#         attention_dropout_rate=dropout_rate,
#         dense_dropout_rate=dropout_rate,
#         norm_output=True,
#     )(x)
#     x = tf.keras.layers.Cropping1D((0, x.shape[1] - 1))(x)
#     x = tf.keras.layers.Reshape([-1])(x)
#
#     x = tf.keras.Sequential(
#         [
#             tf.keras.layers.Dense(ff_dim, activation=gelu),
#             tf.keras.layers.Dense(n_classes, activation="sigmoid"),
#         ],
#         name="mlp_head",
#     )(x)
#
#     model = tf.keras.models.Model([inputs1, inputs2], x)
#     return model
