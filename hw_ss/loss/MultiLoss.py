import torch
from torch import Tensor
from torch import nn
from hw_ss.loss.SISDRLossWrapper import SISPDRLossWrapper
from hw_ss.loss.CELossWrapper import CELossWrapper


class MultiLoss():
    def __init__(self, lambd=0.5, *args, **kwargs):
        self.sisdr = SISPDRLossWrapper(*args, **kwargs)
        self.ce = CELossWrapper()
        self.lambd = lambd

    def forward(self, batch) -> Tensor:
        return self.sisdr(batch) + self.lambd * self.ce(batch)
