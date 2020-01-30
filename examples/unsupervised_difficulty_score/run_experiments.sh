#!/bin/bash

# -----------------------------------------------------------------------------
# Experiments in MNIST
# -----------------------------------------------------------------------------
dataset="MNIST"
validation_split="test"
evaluation_splits="test"

echo "Starting experiments with dataset $dataset and model $model"
for label in 0 1 2 3 4 5 6 7 8 9
do
    echo "Starting experiment $label"
    python3 train.py -d $dataset -v $validation_split -s $evaluation_splits -l $label -m "CNN-KERAS-A"
done 

# -----------------------------------------------------------------------------
# Experiments in CIFAR10
# -----------------------------------------------------------------------------
dataset="CIFAR10"
validation_split="test"
evaluation_splits="test"

echo "Starting experiments with dataset $dataset"
for label in 0 1 2 3 4 5 6 7 8 9
do
    echo "Starting experiments $label"
    python3 train.py -d $dataset -v $validation_split -s $evaluation_splits -l $label -m "XCEPTION-MINI"
    python3 train.py -d $dataset -v $validation_split -s $evaluation_splits -l $label -m "CNN-KERAS-A"
    python3 train.py -d $dataset -v $validation_split -s $evaluation_splits -l $label -m "CNN-KERAS-B"
    python3 train.py -d $dataset -v $validation_split -s $evaluation_splits -l $label -m "RESNET-V2"
done 

# -----------------------------------------------------------------------------
# Experiments in FERPlus
# -----------------------------------------------------------------------------
dataset="FERPlus"
validation_split="val"
evaluation_splits="val"

echo "Starting experiments with dataset $dataset"
for label in 0 1 2 3 4 5 6 7 8 9
do
    echo "Starting experiments $label"
    python3 train.py -d $dataset -v $validation_split -s $evaluation_splits -l $label -m "XCEPTION-MINI"
    python3 train.py -d $dataset -v $validation_split -s $evaluation_splits -l $label -m "CNN-KERAS-A"
    python3 train.py -d $dataset -v $validation_split -s $evaluation_splits -l $label -m "CNN-KERAS-B"
    python3 train.py -d $dataset -v $validation_split -s $evaluation_splits -l $label -m "RESNET-V2"
done 
