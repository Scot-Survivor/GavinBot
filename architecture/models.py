from abc import ABC

import tensorflow as tf
from layers import PositionalEncoding, MultiHeadAttention


class Transformer(tf.keras.Model, ABC):
    """Transformer Model

    Based off paper: https://arxiv.org/pdf/1706.03762.pdf
    ...

    Attributes:
        :arg vocab_size: int
            The vocabulary size the Tokenizer Uses.
        :arg num_layers: int
            The Number of Encoder/Decoder Layers that the model has.
        :arg units: int
            The Number of units the Encoder/Decoder Layers have.
        :arg d_model: int
            Output Units for the Embedding Layers
        :arg num_heads: int
            Number of Heads the MultiHead attention will be configured with
        :arg dropout: float
            The Dropout that the model will have. This number is between 0 and 1. Do not go higher.
        :arg name: str
            The Name the model will be configured with, defaults to "transformer"
        :arg **kwargs
            Key Word arguments to pass to tf.keras.Model super class
    """

    def __init__(self, vocab_size: int, num_layers: int, units: int, d_model: int, num_heads: int, dropout: float,
                 name="transformer", **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self._model_name = name

        inputs = tf.keras.Input(shape=(None,), name="inputs")
        dec_inputs = tf.keras.Input(shape=(None,), name='dec_inputs')

        enc_padding_mask = tf.keras.layers.Lambda(self.create_padding_mask, output_shape=(1, 1, None),
                                                  name="enc_padding_mask")(inputs)

        # mask the future tokens for decoder inputs at the 1st attention block
        look_ahead_mask = tf.keras.layers.Lambda(self.create_look_ahead_mask, output_shape=(1, None, None),
                                                 name="look_ahead_mask")(dec_inputs)

        # mask the encoder outputs for the 2nd attention block
        dec_padding_mask = tf.keras.layers.Lambda(self.create_padding_mask, output_shape=(1, 1, None),
                                                  name="dec_padding_mask")

        enc_outputs = self.encoder()(inputs=[inputs, enc_padding_mask])

        dec_outputs = self.decoder()(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

        outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

        super(Transformer, self).name = name
        super(Transformer, self).inputs = inputs
        super(Transformer, self).outputs = outputs

    def encoder_layer(self, name="encoder_layer"):
        """Encoder Layer
        Arguments:
            :arg name: str
                The name for the layer, returned in model.summary()
        """
        inputs = tf.keras.Input(shape=(None, self.d_model), name='inputs')  # Inputs for the layer
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")  # Input padding mask

        attention = MultiHeadAttention(self.d_model, self.num_heads, name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })  # Attention Layer

        attention = tf.keras.layers.Dropout(rate=self.dropout)(attention)  # Add dropout to the layer
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
            inputs + attention)  # Normalise the inputs and attention values

        # This generates the feedforward part of the encoder
        outputs = tf.keras.layers.Dense(units=self.units, attention='relu')(attention)
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
            attention + outputs)  # Normalise the attention and output values

        return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

    def create_padding_mask(self, x):
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        # batch_size, 1, 1, sequence_length
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, x):
        seq_len = tf.shape(x)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        padding_mask = self.create_padding_mask(x)
        return tf.maximum(look_ahead_mask, padding_mask)

    def encoder(self, name='encoder'):
        """Encoder Sub Model

        Arguments:
            :arg name: str
                The name for the sub model
        """
        inputs = tf.keras.Input(shape=(None,), name='inputs')
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
        embeddings = tf.keras.layers.Embedding(self.vocab_size, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = PositionalEncoding(self.vocab_size, d_model=self.d_model)(
            embeddings)  # Creates the embedding layers

        outputs = tf.keras.layers.Dropout(rate=self.dropout)(embeddings)  # outputs for Encoder sub-model

        for i in range(self.num_heads):  # You know the rules and so do *i*
            outputs = self.encoder_layer(
                name=f"encoding_layer{i}"
            )([outputs, padding_mask])

        return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

    def decoder_layer(self, name="decoder_layer"):
        """Decoder Layer
                Arguments:
                    :arg name: str
                        The name for the layer, returned in model.summary()
                """
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")  # Input layer for decoder layer
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name="encoder_outputs")  # Encoder output
        look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")  # "Look ahead" allows to see ahead of the current word
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")  # Input padding

        attention1 = MultiHeadAttention(self.d_model, self.num_heads, name="attention1")(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
        })  # "map" inputs to inputs through look ahead mask
        attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)  # Normalise the values

        attention2 = MultiHeadAttention(self.d_model, self.num_heads, name="attention2")(inputs={
            'query': attention1,
            'key': enc_outputs,
            'value': enc_outputs,
            'mask': padding_mask
        })  # "Map" attention1 outputs to encoder_outputs through padding mask

        attention2 = tf.keras.layers.Dropout(rate=self.dropout)(attention2)  # Add a dropout to the attention layers
        attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)  # Normalise both attention outputs together

        # Feed Forward network
        outputs = tf.keras.layers.Dense(units=self.units, activation='relu')(attention2)  # attention 1 and 2 were combined prior to this ^
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)  # another dense layer
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(outputs)  # Dropout layer for feed forward
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)  # Normalise before sending to rest of model

        return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)

    def decoder(self, name='decoder'):
        """Decoder Sub Model

        Arguments:
            :arg name: str
                The name for the sub model"""
        inputs = tf.keras.Input(shape=(None,), name='inputs')  # Input for encoder layers
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name='encoder_outputs')  # Output from encoder layers
        look_ahead_mask = tf.keras.Input(shape=(1, 1, None), name='look_ahead_mask')  # Look ahead" allows to see ahead of the current word
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")  # Pad the input
        
        embeddings = tf.keras.layers.Embedding(self.vocab_size, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = PositionalEncoding(self.vocab_size, d_model=self.d_model)(embeddings)

        outputs = tf.keras.layers.Dropout(rate=self.dropout)(embeddings)

        for i in range(self.num_layers):
            outputs = self.decoder_layer(
                name=f'decoder_layer{i}'
            )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

        return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)
