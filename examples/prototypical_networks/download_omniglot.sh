#!/usr/bin/env bash

DATADIR=omniglot/

mkdir -p $DATADIR
wget -O images_background.zip https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip?raw=true
wget -O images_evaluation.zip https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip?raw=true
unzip images_background.zip -d $DATADIR
unzip images_evaluation.zip -d $DATADIR
rm images_background.zip
rm images_evaluation.zip
