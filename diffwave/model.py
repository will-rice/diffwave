"""Model."""
from typing import Optional

import tensorflow as tf

from diffwave import DiffWaveConfig
from diffwave.layers import (
    Conv1D,
    TFDiffusionEmbedding,
    TFResidualBlock,
    TFSpectrogramUpsampler,
)


class TFDiffWave(tf.keras.Model):
    """TFDiffWave Model."""

    def __init__(self, config: DiffWaveConfig):
        super().__init__()
        self._config = config

        self.input_proj = Conv1D(config.residual_channels, 1, activation="relu")
        self.diff_embed = TFDiffusionEmbedding(128)
        self.spec_upsample = TFSpectrogramUpsampler()
        self.res_layers = [
            TFResidualBlock(
                config.residual_channels, 2 ** (i % config.dilation_cycle_length)
            )
            for i in range(config.residual_layers)
        ]
        self.skip_proj = Conv1D(config.residual_channels, 1, activation="relu")
        self.out_proj = Conv1D(1, 1, kernel_initializer="zeros")

    def call(
        self, inputs: tf.Tensor, diff_step: tf.Tensor, cond: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Forward Pass."""
        out = inputs[..., tf.newaxis]
        out = self.input_proj(out)

        diff_step = self.diff_embed(diff_step)

        if cond is not None:
            cond = self.spec_upsample(cond)[:, : tf.shape(out)[1], :]

        skip = 0.0
        for i, layer in enumerate(self.res_layers):
            out, skip = layer(out, diff_step, cond)
            skip += skip

        out = skip * tf.sqrt(self._config.skip_scale)
        out = self.skip_proj(out)
        out = self.out_proj(out)
        return out
