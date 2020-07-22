echo "Creating default directory for dataset ..."
mkdir dataset

echo "Entering new directory ..."
cd dataset/

echo "Using kaggle API to download dataset..."
echo "Make sure to have installed kaggle-API, set-up kaggle API Token and have accepted the rules of the Facial Keypoints Detection challenge"
kaggle competitions download -c facial-keypoints-detection

echo "Unzipping downloaded dataset"
unzip facial-keypoints-detection.zip

echo "Unzipping train split"
unzip training.zip

echo "Unzipping test split"
unzip test.zip

echo "Removing zip files"
rm facial-keypoints-detection.zip
rm training.zip
rm test.zip
