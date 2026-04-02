import numpy as np
import tensorflow as tf
from tensorflow.keras import Model


def get_sinusoidal_encoding(max_len, d_model):
    positions = np.arange(max_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return angles.astype(np.float32)


class SpecAugment(tf.keras.layers.Layer):

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

                         
        for _ in range(self.n_freq_masks):
            f = tf.random.uniform([], 0, self.freq_mask_param, dtype=tf.int32)
            f = tf.minimum(f, features)
            f0 = tf.random.uniform([], 0, features - f, dtype=tf.int32)
            indices = tf.range(features)
            freq_mask = tf.cast(
                tf.logical_or(indices < f0, indices >= f0 + f), x.dtype)
            mask = mask * freq_mask[tf.newaxis, tf.newaxis, :]

                    
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
                                 
        x_norm = self.norm1(x)
        attn_out = self.mha(x_norm, x_norm, training=training)
        x = x + self.dropout1(attn_out, training=training)
                      
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm, training=training)
        x = x + ffn_out
        return x

    def call_training(self, x):
        return self.call(x, training=True)


class TransformerEncoder(Model):

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

                                                      
        if stack_kwargs is not None:
            input_dim = None                                                  
        else:
            input_dim = None
        self.input_proj = tf.keras.layers.Dense(
            d_model,
            kernel_regularizer=tf.keras.regularizers.L2(weightReg))

                             
        if posEncType == 'sinusoidal':
            pe_table = get_sinusoidal_encoding(max_seq_len, d_model)
            self.pos_encoding = tf.constant(pe_table[np.newaxis, :, :])                         
        elif posEncType == 'learned':
            self.pos_embedding = tf.keras.layers.Embedding(max_seq_len, d_model)
        else:
            raise ValueError(f"Unknown posEncType: {posEncType}")

        self.pos_dropout = tf.keras.layers.Dropout(dropout)

                                    
        self.enc_layers = [
            TransformerEncoderLayer(d_model, nhead, d_ff, dropout, attention_dropout)
            for _ in range(num_layers)
        ]

                                                       
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

                                                                               
                                                                                 
        self.dense = tf.keras.layers.Dense(nClasses, dtype='float32')

    def call(self, x, training=False, **kwargs):
                                             
        if self.stack_kwargs is not None:
            x = tf.image.extract_patches(x[:, None, :, :],
                                         sizes=[1, 1, self.stack_kwargs['kernel_size'], 1],
                                         strides=[1, 1, self.stack_kwargs['strides'], 1],
                                         rates=[1, 1, 1, 1],
                                         padding='VALID')
            x = tf.squeeze(x, axis=1)

                            
        x = self.input_proj(x)

                                                                  
                                                                               
        x = x * tf.cast(tf.math.sqrt(float(self.d_model)), x.dtype)

                                 
        seq_len = tf.shape(x)[1]
        if self.posEncType == 'sinusoidal':
            x = x + tf.cast(self.pos_encoding[:, :seq_len, :], x.dtype)
        elif self.posEncType == 'learned':
            positions = tf.range(seq_len)
            x = x + self.pos_embedding(positions)

        x = self.pos_dropout(x, training=training)

                                    
        for layer in self.enc_layers:
            if training and self.gradient_checkpointing:
                x = tf.recompute_grad(layer.call_training)(x)
            else:
                x = layer(x, training=training)

                    
        x = self.final_norm(x)

                                                
        if self.subsampleFactor > 1:
            x = x[:, ::self.subsampleFactor, :]

                           
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

    def __init__(self, n_channels, reduction=8, **kwargs):
        super().__init__(**kwargs)
        bottleneck = max(n_channels // reduction, 1)
        self.fc1 = tf.keras.layers.Dense(bottleneck, activation='relu')
        self.fc2 = tf.keras.layers.Dense(n_channels, activation='sigmoid')

    def call(self, x, training=False):
                                    
                                                
        se = tf.reduce_mean(x, axis=1)                     
                                          
        se = self.fc1(se)
        se = self.fc2(se)                     
                                    
        return x * se[:, tf.newaxis, :]


class SpatialAttention(tf.keras.layers.Layer):

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
        se = tf.reduce_mean(x, axis=1)          
        se = se[:, :, tf.newaxis]             
                                                       
        h = self.proj(se) + self.channel_embed                  
                                      
        attn_out = self.mha(h, h, training=training)                  
        h = self.norm(h + attn_out)
                          
        gate = tf.sigmoid(self.out_proj(h))             
        gate = tf.squeeze(gate, axis=-1)          
                                    
        return x * gate[:, tf.newaxis, :]


class ConformerConvModule(tf.keras.layers.Layer):

    def __init__(self, d_model, conv_kernel_size=31, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pointwise1 = tf.keras.layers.Dense(2 * d_model)           
        self.depthwise = tf.keras.layers.DepthwiseConv1D(
            kernel_size=conv_kernel_size, padding='same', use_bias=False)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.pointwise2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        x = self.norm(x)
        x = self.pointwise1(x)
                                                     
        x1, x2 = tf.split(x, 2, axis=-1)
        x = x1 * tf.sigmoid(x2)
        x = self.depthwise(x)
        x = self.batch_norm(x, training=training)
        x = tf.nn.swish(x)
        x = self.pointwise2(x)
        x = self.dropout(x, training=training)
        return x


class ConformerBlock(tf.keras.layers.Layer):

    def __init__(self, d_model, nhead, d_ff, conv_kernel_size=31,
                 dropout=0.1, attention_dropout=0.0, attn_window=0, **kwargs):
        super().__init__(**kwargs)
        self.attn_window = attn_window
                             
        self.ffn1_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn1 = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='swish'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout),
        ])
                                   
        self.attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead, dropout=attention_dropout)
        self.attn_dropout = tf.keras.layers.Dropout(dropout)
                            
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)
                              
        self.ffn2_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn2 = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='swish'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout),
        ])
                         
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def _make_window_mask(self, seq_len):
        indices = tf.range(seq_len)
                                             
        diff = tf.abs(indices[:, tf.newaxis] - indices[tf.newaxis, :])
                                                                           
        mask = diff <= self.attn_window
                                                                      
        return tf.cast(mask[tf.newaxis, :, :], tf.bool)

    def call(self, x, training=False):
                                  
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x), training=training)
                     
        x_norm = self.attn_norm(x)
        if self.attn_window > 0:
            seq_len = tf.shape(x_norm)[1]
            attn_mask = self._make_window_mask(seq_len)
            attn_out = self.mha(x_norm, x_norm, attention_mask=attn_mask,
                                training=training)
        else:
            attn_out = self.mha(x_norm, x_norm, training=training)
        x = x + self.attn_dropout(attn_out, training=training)
                     
        x = x + self.conv_module(x, training=training)
                                  
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x), training=training)
                         
        x = self.final_norm(x)
        return x

    def call_training(self, x):
        return self.call(x, training=True)


class ConformerEncoder(Model):

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

                                                  
        if squeeze_excitation:
            self.se_block = SqueezeExcitation(se_n_channels, se_reduction)
        else:
            self.se_block = None

                                             
        if spatial_attention:
            self.spatial_attn = SpatialAttention(
                n_channels=se_n_channels, d_attn=spatial_attn_dim,
                nhead=spatial_attn_heads)
        else:
            self.spatial_attn = None

                     
        if spec_augment:
            self.spec_augment = SpecAugment(
                freq_mask_param=freq_mask_param,
                time_mask_param=time_mask_param,
                n_freq_masks=n_freq_masks,
                n_time_masks=n_time_masks)
        else:
            self.spec_augment = None

                          
        self.input_proj = tf.keras.layers.Dense(
            d_model,
            kernel_regularizer=tf.keras.regularizers.L2(weightReg))

                             
        if posEncType == 'sinusoidal':
            pe_table = get_sinusoidal_encoding(max_seq_len, d_model)
            self.pos_encoding = tf.constant(pe_table[np.newaxis, :, :])
        elif posEncType == 'learned':
            self.pos_embedding = tf.keras.layers.Embedding(max_seq_len, d_model)
        else:
            raise ValueError(f"Unknown posEncType: {posEncType}")

        self.pos_dropout = tf.keras.layers.Dropout(dropout)

                          
        self.enc_layers = [
            ConformerBlock(d_model, nhead, d_ff, conv_kernel_size,
                           dropout, attention_dropout, attn_window=attn_window)
            for _ in range(num_layers)
        ]

                                                                             
        self.dense = tf.keras.layers.Dense(nClasses, dtype='float32')

    def call(self, x, training=False, **kwargs):
                                                                   
        if self.se_block is not None:
            x = self.se_block(x, training=training)

                                                            
        if self.spatial_attn is not None:
            x = self.spatial_attn(x, training=training)

                                                   
        if self.stack_kwargs is not None:
            x = tf.image.extract_patches(x[:, None, :, :],
                                         sizes=[1, 1, self.stack_kwargs['kernel_size'], 1],
                                         strides=[1, 1, self.stack_kwargs['strides'], 1],
                                         rates=[1, 1, 1, 1],
                                         padding='VALID')
            x = tf.squeeze(x, axis=1)

                            
        x = self.input_proj(x)

                                
        x = x * tf.cast(tf.math.sqrt(float(self.d_model)), x.dtype)

                                 
        seq_len = tf.shape(x)[1]
        if self.posEncType == 'sinusoidal':
            x = x + tf.cast(self.pos_encoding[:, :seq_len, :], x.dtype)
        elif self.posEncType == 'learned':
            positions = tf.range(seq_len)
            x = x + self.pos_embedding(positions)

        x = self.pos_dropout(x, training=training)

                                     
        if self.spec_augment is not None:
            x = self.spec_augment(x, training=training)

                          
        for layer in self.enc_layers:
            if training and self.gradient_checkpointing:
                x = tf.recompute_grad(layer.call_training)(x)
            else:
                x = layer(x, training=training)

                                   
        if self.subsampleFactor > 1:
            x = x[:, ::self.subsampleFactor, :]

                           
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

                  
    def getIntermediateLayerOutput(self, x):
        x, _ = self.rnn1(x)
        return x

    def getSubsampledTimeSteps(self, timeSteps):
        timeSteps = tf.cast(timeSteps / self.subsampleFactor, dtype=tf.int32)
        if self.stack_kwargs is not None:
            timeSteps = tf.cast((timeSteps - self.stack_kwargs['kernel_size']) / self.stack_kwargs['strides'] + 1, dtype=tf.int32)
        return timeSteps
