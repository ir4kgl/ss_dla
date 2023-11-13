# Speaker separation project

This repository contains speaker separation project done as a part of homework #2 for the DLA course at the CS Faculty of HSE.

## Installation guide


```shell
pip install -r ./requirements.txt
```

## 

## Dataset generation

To train speaker separation model we need to generate train and evaluation data first. You can find dataset generation script in `Dataset_generation.ipynb`

In this notebook we have to write a path to dataset with `*.wav` audios (like [LibriSpeech](https://www.openslr.org/12)) to generate samples from.
Important parameters here are `nfiles` in initialization of `MixtureGenerator` class which represents size of final nixed dataset  and `snr_levels` in call function.

In my experiments for the final train process I used train dataset of size 10000 with and evaluation dataset of size 1000 with `snr_levels=[0, 5]` and `snr_levels=[0]` for train and evaluation splits respectively. The choice of such SNR is based on the experiment settings in the original [SpEx+  paper](https://arxiv.org/pdf/2005.04686.pdf). 

Samples for train are generated from `train-clean-100` [LibriSpeech](https://www.openslr.org/12) dataset split. Samples for evaluation are generated from `test-clean`  [LibriSpeech](https://www.openslr.org/12) dataset split.


## SpEx+ Model

File `hw_ss/model/spex_model.py` contains the implementation of [SpEx+](https://arxiv.org/pdf/2005.04686.pdf) model (with classification head). The code was written by me diligently studying the original article, however I took some inspiration and implementation of auxiliary classes (like `GlobalChannelLayerNorm`) from  [this SpEx+ implementation](https://github.com/gemengtju/SpEx_Plus/blob/master/nnet/conv_tas_net.py).

This model operates as follows: a batch (which is a dictionary with `audio` and `ref` keys required) is passed to forward method, and a dictionary with keys `predicted_audio` and `predicted_logits` is returned.  `predicted_audio` is then being passed to SISDR loss and compared with target audio and `predicted_logits` are passed to Cross-Enthropy loss. 

Note that `predicted_audio` is a tuple of three instances corresponding to short, middle and long decoders.

`predicted_logits` are ignored during evaluation.

## Loss function and metrics

See implementation of the loss function in `hw_ss/loss/MultiLoss.py`, it is namely a weighted sum of SISDR and CE losses. Hyperparameter `lambd` is set to `0.5` to follow the original paper. In SISDR loss (which is a part of `MultiLoss`) we set `alpha=0.1`, `beta=0.1` to follow the SpEx+ paper.

Metrics are `SISDR` and `PESQ`, see the implementation in  `hw_ss/metric/sisdr.py` and  `hw_ss/metric/pesq.py` respectively. Note that we evaluate metrics on short decoder output only.







