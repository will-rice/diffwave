"""Simple dataset."""
import logging
from pathlib import Path
from random import shuffle
from typing import Any, Generator, NamedTuple

import h5py
import librosa
import numpy as np
import tensorflow as tf

from diffwave.config import DiffWaveConfig
from diffwave.typing import TensorLike

LOGGER = logging.getLogger(__name__)


class Sample(NamedTuple):
    """Training Sample"""

    audio: TensorLike
    audio_lengths: TensorLike
    mel_spectrogram: TensorLike
    mel_lengths: TensorLike


class Dataset:
    """Simple dataset."""

    def __init__(self, path: Path, config: DiffWaveConfig) -> None:
        self._path = path
        self._config = config
        self._train: Any = []
        self._val: Any = []
        self._test: Any = []
        self._train_samples: Any = []
        self._val_samples: Any = []
        self._test_samples: Any = []

    def load(self) -> None:

        with open(self._path / "metadata.csv") as file:
            lines = [f.strip() for f in file.readlines()]

        shuffle(lines)

        self._train_samples = lines[: int(0.95 * len(lines))]
        split = lines[len(self._train_samples) :]
        self._val_samples = split[self._config.batch_size :]
        self._test_samples = split[: self._config.batch_size]

        LOGGER.info(f"Train Files: {len(self._train_samples)}")
        LOGGER.info(f"Validation Files: {len(self._val_samples)}")
        LOGGER.info(f"Test Files: {len(self._test_samples)}")

        self._train = self._load_dataset(self._train_samples, slice=True)
        self._val = self._load_dataset(self._val_samples, slice=True)
        self._test = self._load_dataset(self._test_samples)

    def _load_dataset(self, samples: Any, slice: bool = False) -> tf.data.Dataset:
        return (
            tf.data.Dataset.from_generator(
                lambda: self._generate(samples, slice=slice),
                output_signature=Sample(
                    audio=tf.TensorSpec((None,), tf.float32),
                    audio_lengths=tf.TensorSpec((), tf.int32),
                    mel_spectrogram=tf.TensorSpec(
                        (None, self._config.n_mels), tf.float32
                    ),
                    mel_lengths=tf.TensorSpec((), tf.int32),
                ),
            )
            .padded_batch(
                self._config.batch_size,
                padded_shapes=Sample(
                    audio=tf.TensorShape((None,)),
                    audio_lengths=tf.TensorShape(()),
                    mel_spectrogram=tf.TensorShape((None, self._config.n_mels)),
                    mel_lengths=tf.TensorShape(()),
                ),
            )
            .prefetch(tf.data.AUTOTUNE)
        )

    def _generate(
        self, samples: Any, slice: bool = False
    ) -> Generator[Sample, None, None]:
        """Generate a single batch from a collection of samples."""
        for sample in samples:
            filename, _, cleaned = sample.split("|")

            audio, sr = librosa.load(
                self._path / "wavs" / f"{filename}.wav",
                sr=self._config.sample_rate,
            )

            if slice:
                # random sample 1 sec chunks when training
                start: Any = np.random.randint(
                    0, len(audio) - self._config.max_audio_length, size=()
                )
                audio = audio[start : start + self._config.max_audio_length]

            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=self._config.n_mels,
                n_fft=self._config.n_fft,
                hop_length=self._config.hop_length,
                win_length=self._config.window_length,
            )
            mel_spectrogram = librosa.power_to_db(
                mel_spectrogram, ref=np.max
            ).transpose()

            yield Sample(
                audio=audio,
                audio_lengths=len(audio),
                mel_spectrogram=mel_spectrogram,
                mel_lengths=mel_spectrogram.shape[0],
            )

    @property
    def train(self) -> tf.data.Dataset:
        return self._train

    @property
    def test(self) -> tf.data.Dataset:
        return self._test

    @property
    def validate(self) -> tf.data.Dataset:
        return self._val

    @property
    def num_train_steps(self) -> int:
        return len(self._train_samples) // self._config.batch_size

    @property
    def num_test_steps(self) -> int:
        return len(self._test_samples) // self._config.batch_size

    @property
    def num_val_steps(self) -> int:
        return len(self._val_samples) // self._config.batch_size


class HDF5Dataset(Dataset):
    """HDF5Dataset for faster training."""

    def __init__(self, path: Path, config: DiffWaveConfig) -> None:
        super().__init__(path, config)

    def load(self) -> None:
        data_file = h5py.File(self._path, "r")
        samples = list(data_file.values())
        shuffle(samples)

        self._train_samples = samples[: int(0.95 * len(samples))]
        split = samples[len(self._train_samples) :]
        self._val_samples = split[self._config.batch_size :]
        self._test_samples = split[: self._config.batch_size]

        LOGGER.info(f"Train Files: {len(self._train_samples)}")
        LOGGER.info(f"Validation Files: {len(self._val_samples)}")
        LOGGER.info(f"Test Files: {len(self._test_samples)}")

        self._train = self._load_dataset(self._train_samples, slice=True)
        self._val = self._load_dataset(self._val_samples, slice=True)
        self._test = self._load_dataset(self._test_samples)

    def _generate(
        self, samples: Any, slice: bool = False
    ) -> Generator[Sample, None, None]:
        for sample in samples:
            audio = sample["audio"][:]
            audio = audio.astype(np.float32) / (2**15 - 1)

            audio_length = len(audio)
            mel_spectrogram = sample["mel_spectrogram"][:]

            if audio_length < self._config.max_audio_length:
                continue

            if slice:
                # random sample 1 sec chunks when training
                start: Any = np.random.randint(
                    0, audio_length - self._config.max_audio_length, size=()
                )
                audio = audio[start : start + self._config.max_audio_length]

                max_mel_length = (
                    audio_length - self._config.max_audio_length
                ) // self._config.hop_length

                mel_start: Any = start // self._config.hop_length
                mel_spectrogram = mel_spectrogram[
                    mel_start : mel_start + max_mel_length
                ]

            yield Sample(
                audio=audio,
                audio_lengths=len(audio),
                mel_spectrogram=mel_spectrogram,
                mel_lengths=mel_spectrogram.shape[0],
            )
