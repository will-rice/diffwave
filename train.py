import argparse
from pathlib import Path

from diffwave.config import DiffWaveConfig
from diffwave.dataset import Dataset
from diffwave.model import TFDiffWave
from diffwave.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("name", help="name for run")
    parser.add_argument("data_path", type=Path, help="path to dataset")
    parser.add_argument("--log_path", type=Path, help="path to dataset", default="logs")
    args = parser.parse_args()

    config = DiffWaveConfig()
    model = TFDiffWave(config)

    dataset = Dataset(args.data_path, config)

    trainer = Trainer(
        name=args.name,
        config=config,
        model=model,
        dataset=dataset,
        log_path=args.log_path,
    )
    epoch = 0
    while epoch < config.max_epochs:

        trainer.train()
        trainer.test()

        epoch += 1


if __name__ == "__main__":
    main()
