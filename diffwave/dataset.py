"""Simple dataset."""
from pathlib import Path
from random import shuffle
from typing import NamedTuple

import librosa
import numpy as np
import tensorflow as tf

from diffwave.config import DiffWaveConfig
from diffwave.typing import TensorLike


class Sample(NamedTuple):
    audio: TensorLike
    mel_spectrogram: TensorLike


class Dataset:
    def __init__(self, path: Path, config: DiffWaveConfig):
        self._path = path
        self._config = config
        self._train = None
        self._test = None

        with open(self._path / "metadata.csv") as file:
            lines = [f.strip() for f in file.readlines()]

        shuffle(lines)

        train_files = lines[: int(0.8 * len(lines))]
        test_files = lines[: -int(0.2 * len(lines))]

        self._train = self._load(train_files)
        self._test = self._load(test_files)

    def _load(self, samples) -> tf.data.Dataset:
        return (
            tf.data.Dataset.from_generator(
                lambda: self._generate(samples),
                output_signature=Sample(
                    audio=tf.TensorSpec((None,), tf.float32),
                    mel_spectrogram=tf.TensorSpec(
                        (None, self._config.n_mels), tf.float32
                    ),
                ),
            )
            .padded_batch(
                self._config.batch_size,
                padded_shapes=Sample(
                    audio=tf.TensorShape((self._config.max_audio_length,)),
                    mel_spectrogram=tf.TensorShape((None, self._config.n_mels)),
                ),
            )
            .prefetch(tf.data.AUTOTUNE)
        )

    def _generate(self, samples):
        """Generate a single batch from a collection of samples."""
        for sample in samples:

            filename, _, cleaned = sample.split("|")

            audio, sr = librosa.load(
                self._path / "wavs" / f"{filename}.wav",
                sr=self._config.sample_rate,
            )

            # random sample 1 sec chunks
            start = np.random.randint(
                0, len(audio) - self._config.max_audio_length, size=()
            )
            audio = audio[start : start + self._config.max_audio_length]

            spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=self._config.n_mels,
                n_fft=self._config.n_fft,
                hop_length=self._config.hop_length,
                win_length=self._config.window_length,
            )
            log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

            yield Sample(
                audio=audio,
                mel_spectrogram=log_spectrogram.transpose(),
            )

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test
