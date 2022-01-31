# BAI WiSe 21/22

Shrey Dixit, Daniel Speck

## Setup

Using anaconda:

```
conda create --name "bai" python=3.9
conda activate bai
conda install --file requirements.txt
```

## Run Training

```
python train.py hparams/mnist_incremental.json mnist_incremental data
```