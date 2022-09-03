import tensorflow as tf

from diffwave.layers import (
    Conv1D,
    TFDiffusionEmbedding,
    TFResidualBlock,
    TFSpectrogramUpsampler,
)


class TFDiffWave(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self._config = config

        self.input_proj = Conv1D(config.residual_channels, 1, activation="relu")
        self.diff_embed = TFDiffusionEmbedding(128)
        self.spec_upsample = TFSpectrogramUpsampler(config.n_mels)
        self.res_layers = [
            TFResidualBlock(
                config.residual_channels, 2 ** (i % config.dilation_cycle_length)
            )
            for i in range(config.residual_layers)
        ]
        self.skip_proj = Conv1D(config.residual_channels, 1, activation="relu")
        self.out_proj = Conv1D(1, 1, kernel_initializer="zeros")

    def call(self, inputs, diff_step, cond=None):
        out = inputs[..., tf.newaxis]
        out = self.input_proj(out)

        diff_step = self.diff_embed(diff_step)

        if cond is not None:
            cond = self.spec_upsample(cond)[:, : tf.shape(out)[1], :]

        for i, layer in enumerate(self.res_layers):
            out, skip = layer(out, diff_step, cond)

            if i > 0:
                skip += skip

        out = skip / tf.sqrt(float(len(self.res_layers)))
        out = self.skip_proj(out)
        out = self.out_proj(out)
        return out
