import torch
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from hw_ss.base.base_metric import BaseMetric


class SISPDRMetric(BaseMetric):
    def __init__(self, device=torch.device("cuda:0"), alpha=0.1, beta=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.sdisdr = ScaleInvariantSignalDistortionRatio().to(device)

    def __call__(self, batch):
        p_short = batch["predicted_audio"]
        target = batch["target"].squeeze()
        sisdr_short = self.sdisdr.forward(
            preds=p_short,
            target=target
        )
        return sisdr_short
