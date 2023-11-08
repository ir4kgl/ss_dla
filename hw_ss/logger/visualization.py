from enum import Enum

from .wandb import WanDBWriter


class VisualizerBackendType(str, Enum):
    wandb = "wandb"


def get_visualizer(config, logger, backend: VisualizerBackendType):
    return WanDBWriter(config, logger)
