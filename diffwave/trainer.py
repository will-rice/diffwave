"""Simple trainer."""
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from tqdm.auto import tqdm

from diffwave import Dataset, DiffWaveConfig

LOGGER = logging.getLogger(__name__)


class Trainer:
    """Simple Trainer."""

    def __init__(
        self,
        config: DiffWaveConfig,
        model: tf.keras.Model,
        dataset: Dataset,
        log_path: Path,
    ):
        self._config = config
        self._model = model
        self._dataset = dataset
        self._log_path = log_path
        self._loss_fn = tf.keras.losses.MeanAbsoluteError()
        self._train_loss = tf.keras.metrics.Mean()
        self._val_loss = tf.keras.metrics.Mean()
        self._optimizer = tfa.optimizers.MovingAverage(
            tfa.optimizers.AdamW(
                learning_rate=self._config.learning_rate, weight_decay=1e-6
            )
        )

        beta = np.array(self._config.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = tf.convert_to_tensor(noise_level.astype(np.float32))

        wandb.init(
            project=f"diffwave",
            id=log_path.stem,
            name=log_path.stem,
            dir=str(log_path),
        )

        self._ckpt = tf.train.Checkpoint(
            step=tf.Variable(0), optimizer=self._optimizer, net=self._model
        )
        self._ckpt_manager = tf.train.CheckpointManager(
            self._ckpt, log_path, max_to_keep=3
        )
        self._ckpt.restore(self._ckpt_manager.latest_checkpoint)
        if self._ckpt_manager.latest_checkpoint:
            LOGGER.info("Restored from {}".format(self._ckpt_manager.latest_checkpoint))
        else:
            LOGGER.info("Initializing from scratch.")

    def train(self):
        """Train run."""
        with tqdm(total=self._config.steps_per_checkpoint) as pbar:

            for i, batch in enumerate(
                self._dataset.train.take(self._config.steps_per_checkpoint)
            ):
                time_step = tf.random.uniform(
                    (),
                    minval=0,
                    maxval=len(self._config.noise_schedule),
                    dtype=tf.int32,
                )
                noise_scale = self.noise_level[time_step]
                noise_scale_sqrt = noise_scale**0.5
                noise = tf.random.normal(tf.shape(batch.audio))
                noisy_audio = (
                    noise_scale_sqrt * batch.audio + (1.0 - noise_scale) ** 0.5 * noise
                )

                with tf.GradientTape() as tape:
                    outputs = self._model(
                        noisy_audio,
                        diff_step=[time_step],
                        cond=batch.mel_spectrogram,
                        training=True,
                    )

                    loss = self._loss_fn(noise, outputs)
                    self._train_loss.update_state(loss)

                gradients = tape.gradient(loss, self._model.trainable_variables)
                self._optimizer.apply_gradients(
                    zip(gradients, self._model.trainable_variables)
                )

                results = dict(
                    name=self._log_path.stem,
                    train_loss=self._train_loss.result().numpy(),
                )

                pbar.update(i - pbar.n)
                pbar.set_postfix(results)

                results.pop("name")
                wandb.log(results)

                self._ckpt.step.assign_add(1)

    def validate(self):
        """Validate run."""

        with tqdm(total=self._dataset.num_val_steps) as pbar:

            for i, batch in enumerate(self._dataset.validate):

                time_step = tf.random.uniform(
                    (),
                    minval=0,
                    maxval=len(self._config.noise_schedule),
                    dtype=tf.int32,
                )
                noise_scale = self.noise_level[time_step]
                noise_scale_sqrt = noise_scale**0.5
                noise = tf.random.normal(tf.shape(batch.audio))
                noisy_audio = (
                    noise_scale_sqrt * batch.audio + (1.0 - noise_scale) ** 0.5 * noise
                )

                outputs = self._model(
                    noisy_audio,
                    diff_step=[time_step],
                    cond=batch.mel_spectrogram,
                    training=False,
                )

                loss = self._loss_fn(noise, outputs)
                self._val_loss.update_state(loss)

                pbar.update(i - pbar.n)

            results = dict(val_loss=self._val_loss.result().numpy())
            wandb.log(results)

    def test(self):
        """Test run."""

        training_noise_schedule = np.array(self._config.noise_schedule)
        inference_noise_schedule = np.array(self._config.inference_noise_schedule)

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        # TODO: figure out this mess
        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                        talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5
                    )
                    T.append(t + twiddle)
                    break
        T = np.array(T, dtype=np.float32)

        with tqdm(total=self._dataset.num_test_steps) as pbar:

            for i, batch in enumerate(self._dataset.test):

                input_shape = tf.shape(batch.mel_spectrogram)
                audio = tf.random.normal(
                    (input_shape[0], self._config.hop_length * input_shape[1])
                )
                noise_scale = tf.expand_dims(alpha_cum**0.5, 1)

                for n in range(len(alpha) - 1, -1, -1):
                    c1 = 1 / alpha[n] ** 0.5
                    c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
                    outputs = tf.squeeze(
                        self._model(audio, [T[n]], batch.mel_spectrogram), -1
                    )
                    audio = c1 * (audio - c2 * outputs)

                    if n > 0:
                        noise = tf.random.normal((tf.shape(audio)))
                        sigma = (
                            (1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]
                        ) ** 0.5
                        audio += sigma * noise

                    audio = tf.clip_by_value(audio, -1.0, 1.0)

                # log evaluation audio outputs
                silence = np.zeros([int(0.25 * self._config.sample_rate)])
                audio = np.hstack(
                    [
                        np.hstack([o[:l], silence])
                        for o, l in zip(audio, batch.audio_lengths)
                    ]
                )
                wandb.log(
                    {"audio": wandb.Audio(audio, sample_rate=self._config.sample_rate)}
                )
                pbar.update(i - pbar.n)

        save_path = self._ckpt_manager.save()
        LOGGER.info(f"Saved checkpoint for step {int(self._ckpt.step)}: {save_path}")
        self._model.save_weights(str(self._log_path))

    @property
    def model(self):
        """Current model."""
        return self._model

    @property
    def step(self):
        """Global step."""
        return int(self._ckpt.step)
