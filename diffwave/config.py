from dataclasses import dataclass

import numpy as np


@dataclass
class DiffWaveConfig:
    batch_size: int = 2
    learning_rate: float = 2e-4
    clipnorm = 0.0
    max_epochs = 100

    sample_rate = 24000
    n_mels = 80
    n_fft = 1024
    hop_length = 256
    window_length = 1024
    crop_mel_frames = 62

    residual_layers = 30
    residual_channels = 64
    dilation_cycle_length = 10
    noise_schedule = np.linspace(1e-4, 0.05, 50, dtype=np.float32)
    inference_noise_schedule = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5]

    max_audio_length = 16000
