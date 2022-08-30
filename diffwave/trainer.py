"""Simple trainer."""
import numpy as np
import tensorflow as tf
import wandb


class Trainer:
    def __init__(self, name, config, model, dataset, log_path):
        self._config = config
        self._model = model
        self._dataset = dataset
        self._loss_fn = tf.keras.losses.MeanAbsoluteError()
        self._train_loss = tf.keras.metrics.Mean()
        self._test_loss = tf.keras.metrics.Mean()

        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.learning_rate,
            global_clipnorm=self._config.clipnorm,
        )

        wandb.init(project=f"diffwave", id=name, name=name, dir=str(log_path))

        self.noise_level = tf.math.cumprod(1 - self._config.noise_schedule)

    def train(self):

        for i, batch in enumerate(self._dataset.train):

            with tf.GradientTape() as tape:

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
                    training=True,
                )

                loss = self._loss_fn(noise, outputs)
                self._train_loss.update_state(loss)

            grads = tape.gradient(loss, self._model.trainable_variables)
            self._optimizer.apply_gradients(zip(grads, self._model.trainable_weights))

            if i % 200 == 0:
                print(f"step: {i} train_loss: {self._train_loss.result()}")

    def test(self):
        for i, batch in enumerate(self._dataset.test):

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
            self._test_loss.update_state(loss)

            if i % 200 == 0:
                print(f"step: {i} test_loss: {self._test_loss.result()}")
