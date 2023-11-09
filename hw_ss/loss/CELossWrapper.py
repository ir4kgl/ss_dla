import torch
from torch import Tensor
from torch import nn


class CELossWrapper():
    def __init__(self, device=None):
        self.ce = nn.CrossEntropyLoss()

    def forward(self, batch) -> Tensor:
        logits = batch["predicted_logits"]
        return self.ce(logits, batch["speaker_id"])
