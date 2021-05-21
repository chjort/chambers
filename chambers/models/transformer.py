import tensorflow as tf

from chambers.layers.embedding import (
    PositionalEncoding1D,
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
    x_enc = PositionalEncoding1D(embed_dim, name="inputs_positional_encoding")(x_enc)
    x_enc = Encoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=dim_feedforward,
        num_layers=num_encoder_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        pre_norm=False,
    )(x_enc)

    x_dec = tf.keras.layers.Embedding(
        output_vocab_size, embed_dim, mask_zero=True, name="targets_embed"
    )(targets)
    x_dec = PositionalEncoding1D(embed_dim, name="targets_positional_encoding")(x_dec)
    x_dec = Decoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=dim_feedforward,
        num_layers=num_decoder_layers,
        attention_dropout_rate=dropout_rate,
        dense_dropout_rate=dropout_rate,
        pre_norm=False,
        norm_output=False,
        causal=True,
    )([x_dec, x_enc])

    x = tf.keras.layers.Dense(output_vocab_size)(x_dec)

    model = tf.keras.models.Model(inputs=[inputs, targets], outputs=x, name=name)
    return model
