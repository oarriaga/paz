import pytest
from paz.models import SSD300


def test_SSD300_VOC_VOC():
    try:
        model = SSD300(num_classes=21,
                       base_weights='VOC',
                       head_weights='VOC')
    except:
        pytest.fail("SSD VOC-VOC loading failed")


def test_SSD300_VOC_None():
    try:
        model = SSD300(num_classes=2,
                       base_weights='VOC',
                       head_weights=None)
    except:
        pytest.fail("SSD VOC-None loading failed")


def test_SSD300_VGG_None():
    try:
        model = SSD300(num_classes=21,
                       base_weights='VGG',
                       head_weights=None)
    except:
        pytest.fail("SSD VGG-None loading failed")
