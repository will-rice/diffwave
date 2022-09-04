"""Train script."""
import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from diffwave import DiffWaveConfig, HDF5Dataset, TFDiffWave, Trainer


def main():
    """Main entry."""
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("name", help="name for run")
    parser.add_argument("data_path", type=Path, help="path to dataset")
    parser.add_argument("--log_path", type=Path, help="path to dataset", default="logs")
    args = parser.parse_args()

    initialize(1234)

    config = DiffWaveConfig()
    model = TFDiffWave(config)

    dataset = HDF5Dataset(args.data_path, config)
    dataset.load()

    log_path = args.log_path / args.name
    log_path.mkdir(exist_ok=True)

    trainer = Trainer(config=config, model=model, dataset=dataset, log_path=log_path)

    while trainer.step < config.max_steps:

        trainer.train()
        trainer.validate()
        trainer.test()


def initialize(seed: int = 1234) -> None:
    """Set memory growth and seeds."""

    try:
        for d in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(d, True)
    except RuntimeError:
        pass

    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    main()
