import pytest
from paz.models import SSD300


def test_SSD300_VOC_VOC():
    try:
        SSD300(num_classes=21,
               base_weights='VOC',
               head_weights='VOC')
    except ValueError as valuerror:
        pytest.fail("SSD VOC-VOC loading failed: {}". format(valuerror))


def test_SSD300_VOC_None():
    try:
        SSD300(num_classes=2,
               base_weights='VOC',
               head_weights=None)
    except ValueError as valuerror:
        pytest.fail("SSD VOC-None loading failed: {}". format(valuerror))


def test_SSD300_VGG_None():
    try:
        SSD300(num_classes=21,
               base_weights='VGG',
               head_weights=None)
    except ValueError as valuerror:
        pytest.fail("SSD VGG-None loading failed: {}". format(valuerror))
