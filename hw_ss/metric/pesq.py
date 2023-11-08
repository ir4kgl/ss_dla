import torch
from torch import Tensor
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from hw_ss.base.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    def __init__(self, device=torch.device("cuda:0"), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(
            fs=16000, mode='wb').to(device)

    def __call__(self, batch):
        p_short = batch["predicted_audio"][0]
        target = batch["target"].squeeze()
        pesq_short = self.pesq.forward(
            preds=p_short,
            target=target
        )
        return pesq_short
