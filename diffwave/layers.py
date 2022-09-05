"""Model layers."""
from typing import Any, Optional, Tuple, Union

import tensorflow as tf
from keras import layers


class Conv1D(layers.Conv1D):
    """Conv1D layer with explicit padding."""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        padding: Union[int, str, tuple, None] = "valid",
        strides: int = 1,
        dilation_rate: int = 1,
        kernel_initializer: Union[
            tf.keras.initializer, str
        ] = tf.keras.initializers.VarianceScaling(
            scale=1 / 3, mode="fan_in", distribution="uniform"
        ),
        bias_initializer: Union[tf.keras.initializer, str] = "zeros",
        data_format: str = "channels_last",
        **kwargs: Any,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding if isinstance(padding, str) else "valid",
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            data_format=data_format,
            **kwargs,
        )

        if isinstance(padding, tuple):
            self.explicit_padding: Union[tuple, None] = padding
        elif isinstance(padding, int):
            self.explicit_padding = (padding, padding)
        else:
            self.explicit_padding = None

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward Pass."""

        if self.explicit_padding:

            if self.data_format == "channels_first":
                inputs = tf.pad(inputs, ((0, 0), (0, 0), self.explicit_padding))
            else:
                inputs = tf.pad(inputs, ((0, 0), self.explicit_padding, (0, 0)))

        return super().call(inputs)


class SiLU(layers.Layer):
    """SiLU Activation."""

    def call(
        self,
        inputs: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward Pass."""
        return inputs * tf.nn.sigmoid(inputs)


class TFSinusoidalPositionEmbeddings(layers.Layer):
    """Timestep embeddings."""

    def __init__(self, max_steps: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.embeddings = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build embeddings."""
        time = tf.range(self.max_steps)
        half_dim = self.max_steps // 2
        embeddings = tf.math.log(10000.0) / (half_dim - 1)
        embeddings = tf.exp(tf.cast(tf.range(half_dim), tf.float32) * -embeddings)
        embeddings = (
            tf.cast(time[:, tf.newaxis], tf.float32) * embeddings[tf.newaxis, :]
        )
        embeddings = tf.concat(
            (tf.math.sin(embeddings), tf.math.cos(embeddings)), axis=-1
        )
        self.embeddings = tf.Variable(embeddings)

    def call(
        self, time: tf.Tensor, mask: Optional[tf.Tensor] = None, training: bool = False
    ) -> tf.Tensor:
        """Forward Pass."""
        return tf.gather(self.embeddings, time)


class TFDiffusionEmbedding(layers.Layer):
    """Diffusion Embedding Layer."""

    def __init__(self, max_steps: int, units: int = 512, **kwargs: Any):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.time_embed = TFSinusoidalPositionEmbeddings(max_steps)
        self.projection_1 = layers.Dense(units)
        self.activation_1 = SiLU()
        self.projection_2 = layers.Dense(units)
        self.activation_2 = SiLU()

    def lerp(self, step: tf.Tensor) -> tf.Tensor:
        """Interpolate step floor and ceiling."""
        low_idx = tf.math.floor(step)
        high_idx = tf.math.ceil(step)
        low = self.time_embed(tf.cast(low_idx, tf.int32))
        high = self.time_embed(tf.cast(high_idx, tf.int32))
        return low + (high - low) * (step - low_idx)

    def call(
        self, step: tf.Tensor, training: bool = False, mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Forward Pass."""
        if training:
            step = tf.cast(step, tf.int32)
            out = self.time_embed(step)
        else:
            step = tf.cast(step, tf.float32)
            out = self.lerp(step)[tf.newaxis]

        out = self.projection_1(out)
        out = self.activation_1(out)
        out = self.projection_2(out)
        out = self.activation_2(out)
        return out


class TFSpectrogramUpsampler(layers.Layer):
    """Upsample layer for conditional input."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (32, 3),
        strides: Tuple[int, int] = (16, 1),
        padding: str = "same",
        leaky_relu_alpha: float = 0.4,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.conv_1 = layers.Conv2DTranspose(
            1, kernel_size=kernel_size, strides=strides, padding=padding
        )
        self.activation_1 = layers.LeakyReLU(leaky_relu_alpha)
        self.conv_2 = layers.Conv2DTranspose(
            1, kernel_size=kernel_size, strides=strides, padding=padding
        )
        self.activation_2 = layers.LeakyReLU(leaky_relu_alpha)

    def call(
        self,
        inputs: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward Pass."""
        out = tf.expand_dims(inputs, -1)
        out = self.conv_1(out)
        out = self.activation_1(out)
        out = self.conv_2(out)
        out = self.activation_2(out)
        out = tf.squeeze(out, -1)
        return out


class TFResidualBlock(layers.Layer):
    """Residual Block."""

    def __init__(self, residual_channels: int, dilation_rate: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.dilated_conv = Conv1D(
            2 * residual_channels,
            kernel_size=3,
            dilation_rate=dilation_rate,
            padding=dilation_rate,
        )
        self.diff_proj = layers.Dense(residual_channels)
        self.cond_proj = Conv1D(2 * residual_channels, kernel_size=1)
        self.out_proj = Conv1D(2 * residual_channels, kernel_size=1)

    def call(
        self,
        inputs: tf.Tensor,
        diff_step: tf.Tensor,
        cond: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward Pass."""
        diff_step = self.diff_proj(diff_step)
        out = inputs + diff_step
        out = self.dilated_conv(out)

        if cond is not None:
            out = out + self.cond_proj(cond)

        gate, filter_ = tf.split(out, num_or_size_splits=2, axis=-1)
        out = tf.nn.sigmoid(gate) * tf.nn.tanh(filter_)

        out = self.out_proj(out)
        residual, skip = tf.split(out, num_or_size_splits=2, axis=-1)
        out = (inputs + residual) / tf.sqrt(0.5)

        return out, skip
