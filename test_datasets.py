import numpy as np
import torch

import hw_ss.datasets

test_dataset = hw_ss.datasets.MixedLibrispeechDataset(
    "test", "/home/ira/Desktop/DLA/2/ss_dla/dataset_mixes", "/home/ira/Desktop/DLA/2/ss_dla/data")


print(test_dataset.__getitem__(0))

print(test_dataset.__getitem__(0)['audio'].shape)
print(test_dataset.__getitem__(0)['target'].shape)
print(test_dataset.__getitem__(0)['ref'].shape)
