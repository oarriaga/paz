"""RF-DETR / LW-DETR detectors on a windowed DINOv2 backbone."""
from paz.models.detection.rf_detr.models import RFDETRNano
from paz.models.detection.rf_detr.models import RFDETRSmall
from paz.models.detection.rf_detr.models import RFDETRMedium
from paz.models.detection.rf_detr.models import RFDETRBase
from paz.models.detection.rf_detr.models import RFDETRLarge
from paz.models.detection.rf_detr.pretrained import download_weights
