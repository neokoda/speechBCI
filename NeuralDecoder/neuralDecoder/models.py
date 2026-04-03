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


class SpecAugment(tf.keras.layers.Layer):
    """SpecAugment: random time and feature masking for regularization.

    During training, randomly masks contiguous blocks of time steps and
    feature channels. No-op during inference.
    """

    def __init__(self, freq_mask_param=27, time_mask_param=10,
                 n_freq_masks=2, n_time_masks=2, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def call(self, x, training=False):
        if not training:
            return x

        shape = tf.shape(x)
        batch, time_steps, features = shape[0], shape[1], shape[2]

        mask = tf.ones_like(x)

        # Frequency masks
        for _ in range(self.n_freq_masks):
            f = tf.random.uniform([], 0, self.freq_mask_param, dtype=tf.int32)
            f = tf.minimum(f, features)
            f0 = tf.random.uniform([], 0, features - f, dtype=tf.int32)
            indices = tf.range(features)
            freq_mask = tf.cast(
                tf.logical_or(indices < f0, indices >= f0 + f), x.dtype)
            mask = mask * freq_mask[tf.newaxis, tf.newaxis, :]

        # Time masks
        for _ in range(self.n_time_masks):
            t = tf.random.uniform([], 0, self.time_mask_param, dtype=tf.int32)
            t = tf.minimum(t, time_steps)
            t0 = tf.random.uniform([], 0, time_steps - t, dtype=tf.int32)
            indices = tf.range(time_steps)
            time_mask = tf.cast(
                tf.logical_or(indices < t0, indices >= t0 + t), x.dtype)
            mask = mask * time_mask[tf.newaxis, :, tf.newaxis]

        return x * mask


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

    def call_training(self, x):
        """Forward pass with training=True. Used by gradient checkpointing."""
        return self.call(x, training=True)


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
                 max_seq_len=2000,
                 gradient_checkpointing=False):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.subsampleFactor = subsampleFactor
        self.stack_kwargs = stack_kwargs
        self.posEncType = posEncType
        self.gradient_checkpointing = gradient_checkpointing

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

        # Output projection — dtype=float32 ensures logits stay float32 for CTC
        # loss stability even when mixed precision (float16) is enabled globally.
        self.dense = tf.keras.layers.Dense(nClasses, dtype='float32')

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
        # Cast scalar to x.dtype to support mixed precision (float16) training.
        x = x * tf.cast(tf.math.sqrt(float(self.d_model)), x.dtype)

        # Add positional encoding
        seq_len = tf.shape(x)[1]
        if self.posEncType == 'sinusoidal':
            x = x + tf.cast(self.pos_encoding[:, :seq_len, :], x.dtype)
        elif self.posEncType == 'learned':
            positions = tf.range(seq_len)
            x = x + self.pos_embedding(positions)

        x = self.pos_dropout(x, training=training)

        # Transformer encoder layers
        for layer in self.enc_layers:
            if training and self.gradient_checkpointing:
                x = tf.recompute_grad(layer.call_training)(x)
            else:
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


class SqueezeExcitation(tf.keras.layers.Layer):
    """Squeeze-and-Excitation block for channel re-weighting.

    Learns per-sample importance weights for each input channel.
    """

    def __init__(self, n_channels, reduction=8, **kwargs):
        super().__init__(**kwargs)
        bottleneck = max(n_channels // reduction, 1)
        self.fc1 = tf.keras.layers.Dense(bottleneck, activation='relu')
        self.fc2 = tf.keras.layers.Dense(n_channels, activation='sigmoid')

    def call(self, x, training=False):
        # x: (batch, time, channels)
        # Squeeze: global average pool over time
        se = tf.reduce_mean(x, axis=1)  # (batch, channels)
        # Excitation: bottleneck FC layers
        se = self.fc1(se)
        se = self.fc2(se)  # (batch, channels)
        # Scale: broadcast over time
        return x * se[:, tf.newaxis, :]


class SpatialAttention(tf.keras.layers.Layer):
    """Cross-channel self-attention for electrode spatial relationships.

    Like SE but replaces the FC bottleneck with multi-head self-attention
    across channels. Each channel selectively attends to specific other
    channels (content-dependent) rather than mixing all channels equally
    through a shared bottleneck.

    Learnable channel embeddings give the attention mechanism knowledge
    of electrode identity, enabling position-specific spatial patterns.
    """

    def __init__(self, n_channels=256, d_attn=64, nhead=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.d_attn = d_attn
        self.proj = tf.keras.layers.Dense(d_attn)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_attn // nhead, dropout=dropout)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.out_proj = tf.keras.layers.Dense(1)

    def build(self, input_shape):
        self.channel_embed = self.add_weight(
            'channel_embed',
            shape=(1, self.n_channels, self.d_attn),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True)
        super().build(input_shape)

    def call(self, x, training=False):
        # x: (B, T, C) where C = n_channels = 256
        # Squeeze: global average pool over time
        se = tf.reduce_mean(x, axis=1)  # (B, C)
        se = se[:, :, tf.newaxis]  # (B, C, 1)
        # Project to attention space + channel identity
        h = self.proj(se) + self.channel_embed  # (B, C, d_attn)
        # Cross-channel self-attention
        attn_out = self.mha(h, h, training=training)  # (B, C, d_attn)
        h = self.norm(h + attn_out)
        # Per-channel gate
        gate = tf.sigmoid(self.out_proj(h))  # (B, C, 1)
        gate = tf.squeeze(gate, axis=-1)  # (B, C)
        # Scale: broadcast over time
        return x * gate[:, tf.newaxis, :]


class ConformerConvModule(tf.keras.layers.Layer):
    """Conformer convolution module: LayerNorm → Pointwise Conv → GLU →
    Depthwise Conv → BatchNorm → Swish → Pointwise Conv → Dropout."""

    def __init__(self, d_model, conv_kernel_size=31, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pointwise1 = tf.keras.layers.Dense(2 * d_model)  # for GLU
        self.depthwise = tf.keras.layers.DepthwiseConv1D(
            kernel_size=conv_kernel_size, padding='same', use_bias=False)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.pointwise2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        x = self.norm(x)
        x = self.pointwise1(x)
        # GLU activation: split in half, sigmoid gate
        x1, x2 = tf.split(x, 2, axis=-1)
        x = x1 * tf.sigmoid(x2)
        x = self.depthwise(x)
        x = self.batch_norm(x, training=training)
        x = tf.nn.swish(x)
        x = self.pointwise2(x)
        x = self.dropout(x, training=training)
        return x


class ConformerBlock(tf.keras.layers.Layer):
    """Single Conformer block: FFN(½) → MHSA → Conv → FFN(½) → LayerNorm."""

    def __init__(self, d_model, nhead, d_ff, conv_kernel_size=31,
                 dropout=0.1, attention_dropout=0.0, attn_window=0, **kwargs):
        super().__init__(**kwargs)
        self.attn_window = attn_window
        # First half-step FFN
        self.ffn1_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn1 = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='swish'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout),
        ])
        # Multi-head self-attention
        self.attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead, dropout=attention_dropout)
        self.attn_dropout = tf.keras.layers.Dropout(dropout)
        # Convolution module
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)
        # Second half-step FFN
        self.ffn2_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn2 = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='swish'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout),
        ])
        # Final LayerNorm
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def _make_window_mask(self, seq_len):
        """Create a band-diagonal attention mask allowing each position
        to attend only within +/- attn_window positions."""
        indices = tf.range(seq_len)
        # (T, T) matrix of absolute distances
        diff = tf.abs(indices[:, tf.newaxis] - indices[tf.newaxis, :])
        # True where within window (allowed), False where outside (blocked)
        mask = diff <= self.attn_window
        # Keras MHA expects (B, T, T) or broadcastable — use (1, T, T)
        return tf.cast(mask[tf.newaxis, :, :], tf.bool)

    def call(self, x, training=False):
        # FFN module 1 (half-step)
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x), training=training)
        # MHSA module
        x_norm = self.attn_norm(x)
        if self.attn_window > 0:
            seq_len = tf.shape(x_norm)[1]
            attn_mask = self._make_window_mask(seq_len)
            attn_out = self.mha(x_norm, x_norm, attention_mask=attn_mask,
                                training=training)
        else:
            attn_out = self.mha(x_norm, x_norm, training=training)
        x = x + self.attn_dropout(attn_out, training=training)
        # Conv module
        x = x + self.conv_module(x, training=training)
        # FFN module 2 (half-step)
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x), training=training)
        # Final LayerNorm
        x = self.final_norm(x)
        return x

    def call_training(self, x):
        """Forward pass with training=True. Used by gradient checkpointing."""
        return self.call(x, training=True)


class ConformerEncoder(Model):
    """Conformer encoder for sequence-to-sequence prediction with CTC.

    Drop-in replacement for TransformerEncoder with identical API.
    Adds convolution modules inside each layer for local feature extraction.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 num_layers,
                 d_ff,
                 nClasses,
                 conv_kernel_size=31,
                 weightReg=1e-5,
                 dropout=0.1,
                 attention_dropout=0.0,
                 posEncType='sinusoidal',
                 subsampleFactor=1,
                 stack_kwargs=None,
                 max_seq_len=2000,
                 gradient_checkpointing=False,
                 spec_augment=False,
                 freq_mask_param=27,
                 time_mask_param=10,
                 n_freq_masks=2,
                 n_time_masks=2,
                 squeeze_excitation=False,
                 se_n_channels=256,
                 se_reduction=8,
                 spatial_attention=False,
                 spatial_attn_dim=64,
                 spatial_attn_heads=4,
                 attn_window=0):
        super(ConformerEncoder, self).__init__()

        self.d_model = d_model
        self.subsampleFactor = subsampleFactor
        self.stack_kwargs = stack_kwargs
        self.posEncType = posEncType
        self.gradient_checkpointing = gradient_checkpointing

        # Squeeze-and-Excitation (before stacking)
        if squeeze_excitation:
            self.se_block = SqueezeExcitation(se_n_channels, se_reduction)
        else:
            self.se_block = None

        # Spatial attention (before stacking)
        if spatial_attention:
            self.spatial_attn = SpatialAttention(
                n_channels=se_n_channels, d_attn=spatial_attn_dim,
                nhead=spatial_attn_heads)
        else:
            self.spatial_attn = None

        # SpecAugment
        if spec_augment:
            self.spec_augment = SpecAugment(
                freq_mask_param=freq_mask_param,
                time_mask_param=time_mask_param,
                n_freq_masks=n_freq_masks,
                n_time_masks=n_time_masks)
        else:
            self.spec_augment = None

        # Input projection
        self.input_proj = tf.keras.layers.Dense(
            d_model,
            kernel_regularizer=tf.keras.regularizers.L2(weightReg))

        # Positional encoding
        if posEncType == 'sinusoidal':
            pe_table = get_sinusoidal_encoding(max_seq_len, d_model)
            self.pos_encoding = tf.constant(pe_table[np.newaxis, :, :])
        elif posEncType == 'learned':
            self.pos_embedding = tf.keras.layers.Embedding(max_seq_len, d_model)
        else:
            raise ValueError(f"Unknown posEncType: {posEncType}")

        self.pos_dropout = tf.keras.layers.Dropout(dropout)

        # Conformer blocks
        self.enc_layers = [
            ConformerBlock(d_model, nhead, d_ff, conv_kernel_size,
                           dropout, attention_dropout, attn_window=attn_window)
            for _ in range(num_layers)
        ]

        # Output projection — float32 for CTC stability under mixed precision
        self.dense = tf.keras.layers.Dense(nClasses, dtype='float32')

    def call(self, x, training=False, **kwargs):
        # Squeeze-and-Excitation (on raw channels, before stacking)
        if self.se_block is not None:
            x = self.se_block(x, training=training)

        # Spatial attention (cross-channel, before stacking)
        if self.spatial_attn is not None:
            x = self.spatial_attn(x, training=training)

        # Stack/subsample (same as Transformer/GRU)
        if self.stack_kwargs is not None:
            x = tf.image.extract_patches(x[:, None, :, :],
                                         sizes=[1, 1, self.stack_kwargs['kernel_size'], 1],
                                         strides=[1, 1, self.stack_kwargs['strides'], 1],
                                         rates=[1, 1, 1, 1],
                                         padding='VALID')
            x = tf.squeeze(x, axis=1)

        # Project to d_model
        x = self.input_proj(x)

        # Scale by sqrt(d_model)
        x = x * tf.cast(tf.math.sqrt(float(self.d_model)), x.dtype)

        # Add positional encoding
        seq_len = tf.shape(x)[1]
        if self.posEncType == 'sinusoidal':
            x = x + tf.cast(self.pos_encoding[:, :seq_len, :], x.dtype)
        elif self.posEncType == 'learned':
            positions = tf.range(seq_len)
            x = x + self.pos_embedding(positions)

        x = self.pos_dropout(x, training=training)

        # SpecAugment (training only)
        if self.spec_augment is not None:
            x = self.spec_augment(x, training=training)

        # Conformer blocks
        for layer in self.enc_layers:
            if training and self.gradient_checkpointing:
                x = tf.recompute_grad(layer.call_training)(x)
            else:
                x = layer(x, training=training)

        # Subsample (if applicable)
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
