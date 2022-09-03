# LRP Pruning

## Copy From
>  https://github.com/seulkiyeom/

## Requirements
> Pytorch 1.10\
> others

## Model
> VGG-16, AlexNet

## Usage

understand the pruning progress.

prepare your dataset. see prune_vgg.py setup_dataloaders

retrain your model. use para --train

prune by choices below:
> 1. run main.py with params, see get_args
> 2. run main.py with config.json ,see config.json
> 3. generate bash.sh , run bash

analysis result. using analysis.py

#### Windows pycharm bash
> setting -> terminal = cmd \
> terminal (your virtual env) : *.sh



