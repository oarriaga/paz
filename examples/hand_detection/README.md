# Hand detection example
In this example we fine-tune a pre-trained object detector in the [Hand Dataset](https://www.robots.ox.ac.uk/~vgg/data/hands/).

## Downloading partial openImagesV6 dataset
We use the recommended tool for downloading specific classes from the OpenImagesV6 dataset.

This requires to install the fiftyone download API:
```
pip install --user fiftyone 
```

Now you can call this simple script to download the partial set for hands
```python
python download_openimagesV6.py
```

Maybe optional?
Installing fiftyone pypi package messed up my openCV installation thus I had to remove their openCV installation
```
pip uninstall opencv-python-headless
```


## TODO
- Update prior boxes with training bounding boxes

SSD512 + unfrozen base (+updated prior boxes) with a batch size 32 on OpenImagesV6 dataset


