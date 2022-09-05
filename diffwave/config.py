"""Model Configuration."""
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np


@dataclass
class DiffWaveConfig:
    """Diffwave Model Configuration."""

    batch_size: int = 16
    learning_rate: float = 2e-4
    clipnorm: Union[None, float] = None
    steps_per_checkpoint: int = 1000
    max_steps: int = 500000

    sample_rate: int = 24000
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    window_length: int = 1024

    residual_layers: int = 30
    residual_channels: int = 64
    dilation_cycle_length: int = 10
    skip_scale = 1.0 / float(residual_layers)
    noise_schedule: np.ndarray = np.linspace(1e-4, 0.05, 50, dtype=np.float32)
    inference_noise_schedule: Tuple[float, ...] = (0.001, 0.01, 0.05, 0.2, 0.5)

    max_audio_length: int = 8192
