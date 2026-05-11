"""CLI entry point for a single training run."""

from __future__ import annotations

import argparse
import logging
from dataclasses import fields

from gen_cats.config import TrainConfig
from gen_cats.factory import create_dataloaders, create_trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a generative model")

    for f in fields(TrainConfig):
        flag = f"--{f.name.replace('_', '-')}"
        if f.type == "bool":
            parser.add_argument(flag, type=lambda x: x.lower() in ("true", "1", "yes"))
        elif "float" in str(f.type):
            parser.add_argument(flag, type=float, default=None)
        elif "int" in str(f.type):
            parser.add_argument(flag, type=int, default=None)
        else:
            parser.add_argument(flag, type=str, default=None)

    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    return TrainConfig(**overrides)


def main() -> None:
    config = parse_args()
    train_loader, val_loader = create_dataloaders(config)
    trainer = create_trainer(config)

    logging.getLogger(__name__).info(
        "Starting %s training (seed=%d, epochs=%d)",
        config.model_type,
        config.seed,
        config.max_epochs,
    )

    results = trainer.fit(train_loader, val_loader)
    print(f"Training complete: {results}")


if __name__ == "__main__":
    main()
