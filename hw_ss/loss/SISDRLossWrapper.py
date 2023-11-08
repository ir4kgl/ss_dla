import torch
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SISPDRLossWrapper():
    def __init__(self, alpha=0.1, beta=0.1):
        self.alpha = alpha
        self.beta = beta
        self.sdisdr = ScaleInvariantSignalDistortionRatio()

    def forward(self, batch) -> Tensor:
        p_short, p_middle, p_long = batch["predicted_audio"]
        target = batch["target"].squeeze()
        sisdr_short = self.sdisdr.forward(
            preds=p_short,
            target=target
        )
        sisdr_middle = self.sdisdr.forward(
            preds=p_middle,
            target=target
        )
        sisdr_long = self.sdisdr.forward(
            preds=p_long,
            target=target
        )
        return -(1 - self.alpha - self.beta) * sisdr_short + self.alpha * sisdr_middle + self.beta * sisdr_long
