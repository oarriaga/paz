#!/usr/bin/env bash
# Downloads the Kaggle "Facial Keypoints Detection" dataset into dataset/.
# Requires the kaggle CLI, a configured API token, and accepting the
# competition rules (otherwise the API returns 403).
set -e
mkdir -p dataset
cd dataset/
kaggle competitions download -c facial-keypoints-detection
unzip -o facial-keypoints-detection.zip
unzip -o training.zip
unzip -o test.zip
rm facial-keypoints-detection.zip training.zip test.zip
