import argparse
import os
import sys
import warnings

from omegaconf import OmegaConf
import wandb

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from liveworld.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--disable_wandb", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    # Merge default config if specified
    default_config_path = config.get("default_config_path", None)
    if default_config_path and os.path.exists(default_config_path):
        default_config = OmegaConf.load(default_config_path)
        config = OmegaConf.merge(default_config, config)

    config.config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_path = args.config_path
    config.disable_wandb = args.disable_wandb

    trainer = Trainer(config)
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
