import tensorflow as tf
from keras import layers


class SiLU(layers.Layer):
    def call(self, inputs):
        return inputs * tf.nn.sigmoid(inputs)


class TFDiffusionEmbedding(layers.Layer):
    def __init__(self, max_steps, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.projection_1 = layers.Dense(512)
        self.activation_1 = SiLU()
        self.projection_2 = layers.Dense(512)
        self.activation_2 = SiLU()

    def build(self, input_shape):
        steps = tf.expand_dims(tf.range(self.max_steps, dtype=tf.float32), 1)
        dims = tf.range(64, dtype=tf.float32)[tf.newaxis]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)
        table = tf.concat([tf.math.sin(table), tf.math.cos(table)], axis=1)
        self.embedding = tf.Variable(table, dtype=tf.float32)
        super().build(input_shape)

    def lerp(self, step):
        low_idx = tf.math.floor(step)
        high_idx = tf.math.ceil(step)
        low = tf.gather(self.embedding, int(low_idx))
        high = tf.gather(self.embedding, int(high_idx))
        return low + (high - low) * (step - low_idx)

    def call(self, step, training=False, mask=None):

        if training:
            step = tf.cast(step, tf.int32)
            out = tf.gather(self.embedding, step)
        else:
            out = self.lerp(step)[tf.newaxis]

        out = self.projection_1(out)
        out = self.activation_1(out)
        out = self.projection_2(out)
        out = self.activation_2(out)
        return out


class TFSpectrogramUpsampler(layers.Layer):
    def __init__(self, n_mels, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = layers.Conv2DTranspose(
            1, kernel_size=(32, 3), strides=(16, 1), padding="same"
        )
        self.activation_1 = layers.LeakyReLU(0.4)
        self.conv_2 = layers.Conv2DTranspose(
            1, kernel_size=(32, 3), strides=(16, 1), padding="same"
        )
        self.activation_2 = layers.LeakyReLU(0.4)

    def call(self, inputs):
        out = tf.expand_dims(inputs, -1)
        out = self.conv_1(out)
        out = self.activation_1(out)
        out = self.conv_2(out)
        out = self.activation_2(out)
        out = tf.squeeze(out, -1)
        return out


class TFResidualBlock(layers.Layer):
    def __init__(self, num_mels, residual_channels, dilation_rate, **kwargs):
        super().__init__(**kwargs)
        self.dilated_conv = layers.Conv1D(
            2 * residual_channels,
            kernel_size=3,
            dilation_rate=dilation_rate,
            padding="same",
        )
        self.diff_proj = layers.Dense(residual_channels)
        self.cond_proj = layers.Conv1D(2 * residual_channels, kernel_size=1)
        self.out_proj = layers.Conv1D(2 * residual_channels, kernel_size=1)

    def call(self, inputs, diff_step, cond=None):
        diff_step = tf.expand_dims(self.diff_proj(diff_step), 1)
        out = inputs + diff_step
        out = self.dilated_conv(out)

        if cond is not None:
            out = out + self.cond_proj(cond)

        gate, filter_ = tf.split(out, num_or_size_splits=2, axis=-1)
        out = tf.nn.sigmoid(gate) * tf.nn.tanh(filter_)

        out = self.out_proj(out)
        residual, skip = tf.split(out, num_or_size_splits=2, axis=-1)
        out = (inputs + residual) / tf.sqrt(2.0)

        return out, skip
