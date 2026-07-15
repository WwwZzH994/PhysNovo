# PhysNovo

This repository contains inference code for **PhysNovo**.

## Architecture

The overall architecture of PhysNovo.

![model](model.pdf)

## Requirements

Set up the required software environment and install all dependencies before training or inference.

```
conda env create -f environment.yml
conda activate physnovo
```

## Datasets

Download the datasets used for model training and evaluation.

- Nine-species benchmark dataset ：https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000081382
- Aggregated data：https://zenodo.org/records/10405582
- Non-Enzymatic data：https://doi.org/10.25345/C5KS6JG0W


## Pretrained Models

Download the pretrained PhysNovo models corresponding to different datasets for validation or inference.

- Nine-Species Model：https://doi.org/10.5281/zenodo.20687570
- Aggregated Model：https://doi.org/10.5281/zenodo.21355375
- Non-Enzymatic Model：https://doi.org/10.5281/zenodo.20687690

## Train

Train PhysNovo on your own dataset by specifying the training and validation spectrum files.

```
python main.py --mode=train --gpu=0,1 --config=./config.yaml --output=train.log --peak_path=./*.mgf --peak_path_val=./*.mgf
```

## Validation

Evaluate a pretrained PhysNovo model on a test dataset by providing the model checkpoint and spectrum file.

```
python main.py --mode=eval --gpu=0,1 --config=./config.yaml --output=evaluate.log --peak_path=./*.mgf --model=the_path_of_your_model
```




