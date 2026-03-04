import numpy as np
import tensorflow as tf
from tensorflow.keras import Model


def get_sinusoidal_encoding(max_len, d_model):
    """Generate sinusoidal positional encoding table."""
    positions = np.arange(max_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return angles.astype(np.float32)


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Single Transformer encoder layer with pre-LayerNorm."""

    def __init__(self, d_model, nhead, d_ff, dropout=0.1, attention_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead, dropout=attention_dropout)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout),
        ])
        self.dropout1 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out = self.mha(x_norm, x_norm, training=training)
        x = x + self.dropout1(attn_out, training=training)
        # Pre-norm FFN
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm, training=training)
        x = x + ffn_out
        return x


class TransformerEncoder(Model):
    """Transformer encoder for sequence-to-sequence prediction with CTC.

    Drop-in replacement for the GRU model with identical API:
      - call(x, training=False) → logits (B, T', nClasses)
      - getSubsampledTimeSteps(timeSteps) for CTC
    """

    def __init__(self,
                 d_model,
                 nhead,
                 num_layers,
                 d_ff,
                 nClasses,
                 weightReg=1e-5,
                 dropout=0.1,
                 attention_dropout=0.0,
                 posEncType='sinusoidal',
                 subsampleFactor=1,
                 stack_kwargs=None,
                 max_seq_len=2000):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.subsampleFactor = subsampleFactor
        self.stack_kwargs = stack_kwargs
        self.posEncType = posEncType

        # Input projection: stack output dim → d_model
        if stack_kwargs is not None:
            input_dim = None  # Determined dynamically based on input features
        else:
            input_dim = None
        self.input_proj = tf.keras.layers.Dense(
            d_model,
            kernel_regularizer=tf.keras.regularizers.L2(weightReg))

        # Positional encoding
        if posEncType == 'sinusoidal':
            pe_table = get_sinusoidal_encoding(max_seq_len, d_model)
            self.pos_encoding = tf.constant(pe_table[np.newaxis, :, :])  # (1, max_len, d_model)
        elif posEncType == 'learned':
            self.pos_embedding = tf.keras.layers.Embedding(max_seq_len, d_model)
        else:
            raise ValueError(f"Unknown posEncType: {posEncType}")

        self.pos_dropout = tf.keras.layers.Dropout(dropout)

        # Transformer encoder layers
        self.enc_layers = [
            TransformerEncoderLayer(d_model, nhead, d_ff, dropout, attention_dropout)
            for _ in range(num_layers)
        ]

        # Final norm (needed for pre-norm architecture)
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Output projection
        self.dense = tf.keras.layers.Dense(nClasses)

    def call(self, x, training=False, **kwargs):
        # Stack/subsample (same logic as GRU)
        if self.stack_kwargs is not None:
            x = tf.image.extract_patches(x[:, None, :, :],
                                         sizes=[1, 1, self.stack_kwargs['kernel_size'], 1],
                                         strides=[1, 1, self.stack_kwargs['strides'], 1],
                                         rates=[1, 1, 1, 1],
                                         padding='VALID')
            x = tf.squeeze(x, axis=1)

        # Project to d_model
        x = self.input_proj(x)

        # Scale by sqrt(d_model) as in "Attention Is All You Need"
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Add positional encoding
        seq_len = tf.shape(x)[1]
        if self.posEncType == 'sinusoidal':
            x = x + self.pos_encoding[:, :seq_len, :]
        elif self.posEncType == 'learned':
            positions = tf.range(seq_len)
            x = x + self.pos_embedding(positions)

        x = self.pos_dropout(x, training=training)

        # Transformer encoder layers
        for layer in self.enc_layers:
            x = layer(x, training=training)

        # Final norm
        x = self.final_norm(x)

        # Subsample (if applicable, same as GRU)
        if self.subsampleFactor > 1:
            x = x[:, ::self.subsampleFactor, :]

        # Output projection
        x = self.dense(x, training=training)
        return x

    def getSubsampledTimeSteps(self, timeSteps):
        timeSteps = tf.cast(timeSteps / self.subsampleFactor, dtype=tf.int32)
        if self.stack_kwargs is not None:
            timeSteps = tf.cast(
                (timeSteps - self.stack_kwargs['kernel_size']) / self.stack_kwargs['strides'] + 1,
                dtype=tf.int32)
        return timeSteps


class GRU(Model):
    def __init__(self,
                 units,
                 weightReg,
                 actReg,
                 subsampleFactor,
                 nClasses,
                 bidirectional=False,
                 dropout=0.0,
                 nLayers=2,
                 conv_kwargs=None,
                 stack_kwargs=None):
        super(GRU, self).__init__()

        weightReg = tf.keras.regularizers.L2(weightReg)
        #actReg = tf.keras.regularizers.L2(actReg)
        actReg = None
        recurrent_init = tf.keras.initializers.Orthogonal()
        kernel_init = tf.keras.initializers.glorot_uniform()
        self.subsampleFactor = subsampleFactor
        self.bidirectional = bidirectional
        self.stack_kwargs = stack_kwargs

        if bidirectional:
            self.initStates = [
                tf.Variable(initial_value=kernel_init(shape=(1, units))),
                tf.Variable(initial_value=kernel_init(shape=(1, units))),
            ]
        else:
            self.initStates = tf.Variable(initial_value=kernel_init(shape=(1, units)))

        self.conv1 = None
        if conv_kwargs is not None:
            self.conv1 = tf.keras.layers.DepthwiseConv1D(
                                                **conv_kwargs,
                                               padding='same',
                                               activation='relu',
                                               kernel_regularizer=weightReg,
                                               use_bias=False)

        self.rnnLayers = []
        for _ in range(nLayers):
            rnn = tf.keras.layers.GRU(units,
                                      return_sequences=True,
                                      return_state=True,
                                      kernel_regularizer=weightReg,
                                      activity_regularizer=actReg,
                                      recurrent_initializer=recurrent_init,
                                      kernel_initializer=kernel_init,
                                      dropout=dropout)
            self.rnnLayers.append(rnn)
        if bidirectional:
            self.rnnLayers = [tf.keras.layers.Bidirectional(rnn) for rnn in self.rnnLayers]
        self.dense = tf.keras.layers.Dense(nClasses)

    def call(self, x, states=None, training=False, returnState=False):
        batchSize = tf.shape(x)[0]

        if self.stack_kwargs is not None:
            x = tf.image.extract_patches(x[:, None, :, :],
                                         sizes=[1, 1, self.stack_kwargs['kernel_size'], 1],
                                         strides=[1, 1, self.stack_kwargs['strides'], 1],
                                         rates=[1, 1, 1, 1],
                                         padding='VALID')
            x = tf.squeeze(x, axis=1)

        if self.conv1 is not None:
            x = self.conv1(x)

        if states is None:
            states = []
            if self.bidirectional:
                states.append([tf.tile(s, [batchSize, 1]) for s in self.initStates])
            else:
                states.append(tf.tile(self.initStates, [batchSize, 1]))
            states.extend([None] * (len(self.rnnLayers) - 1))

        new_states = []
        if self.bidirectional:
            for i, rnn in enumerate(self.rnnLayers):
                x, forward_s, backward_s = rnn(x, training=training, initial_state=states[i])
                if i == len(self.rnnLayers) - 2:
                    if self.subsampleFactor > 1:
                        x = x[:, ::self.subsampleFactor, :]
                new_states.append([forward_s, backward_s])
        else:
            for i, rnn in enumerate(self.rnnLayers):
                x, s = rnn(x, training=training, initial_state=states[i])
                if i == len(self.rnnLayers) - 2:
                    if self.subsampleFactor > 1:
                        x = x[:, ::self.subsampleFactor, :]
                new_states.append(s)

        x = self.dense(x, training=training)

        if returnState:
            return x, new_states
        else:
            return x

    # TODO: Fix me
    def getIntermediateLayerOutput(self, x):
        x, _ = self.rnn1(x)
        return x

    def getSubsampledTimeSteps(self, timeSteps):
        timeSteps = tf.cast(timeSteps / self.subsampleFactor, dtype=tf.int32)
        if self.stack_kwargs is not None:
            timeSteps = tf.cast((timeSteps - self.stack_kwargs['kernel_size']) / self.stack_kwargs['strides'] + 1, dtype=tf.int32)
        return timeSteps
