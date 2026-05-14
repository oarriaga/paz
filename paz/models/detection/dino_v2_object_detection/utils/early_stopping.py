import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EarlyStoppingCallback:
    """Early stopping callback that monitors mAP.

    Stops training when mAP has not improved by at least *min_delta*
    for *patience* consecutive epochs.

    Attributes:
        patience (int): Epochs without improvement before stopping.
        min_delta (float): Minimum mAP improvement to reset the counter.
        use_ema (bool): If True, prefer EMA model metrics.
        verbose (bool): Print status messages each epoch.
        best_map (float): Best mAP observed so far.
        counter (int): Consecutive epochs without improvement.
        model: Keras model whose ``stop_training`` flag is set.
        segmentation_head (bool): If True, monitor mask metrics instead
            of bounding-box metrics.
    """

    def __init__(
        self,
        model: Any,
        patience: int = 5,
        min_delta: float = 0.001,
        use_ema: bool = False,
        verbose: bool = True,
        segmentation_head: bool = False,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.use_ema = use_ema
        self.verbose = verbose
        self.best_map = 0.0
        self.counter = 0
        self.model = model
        self.segmentation_head = segmentation_head

    def update(self, log_stats: Dict[str, Any]) -> None:
        """Update early stopping state from epoch validation metrics.

        Extracts mAP from *log_stats* (using bounding-box or mask
        evaluation keys depending on ``segmentation_head``), compares
        against the best recorded mAP, and increments or resets the
        patience counter accordingly.

        Args:
            log_stats (dict): Epoch log dictionary, expected to contain
                one or more of ``test_coco_eval_bbox``,
                ``ema_test_coco_eval_bbox``,
                ``test_coco_eval_masks``, or
                ``ema_test_coco_eval_masks``.
        """
        regular_map = None
        ema_map = None

        # --- Extract regular (non-EMA) mAP ---
        if 'test_coco_eval_bbox' in log_stats:
            val = log_stats['test_coco_eval_bbox']
            # Handle if val is list or single value
            if isinstance(val, (list, tuple)) and len(val) > 0:
                 if not self.segmentation_head:
                    regular_map = val[0]
            elif isinstance(val, (float, int)):
                 regular_map = val
            
        # If a segmentation head is used, override with mask metrics
        if 'test_coco_eval_masks' in log_stats and self.segmentation_head:
            val = log_stats['test_coco_eval_masks']
            if isinstance(val, (list, tuple)) and len(val) > 0:
                regular_map = val[0]
            elif isinstance(val, (float, int)):
                regular_map = val

        # --- Extract EMA mAP ---
        if 'ema_test_coco_eval_bbox' in log_stats:
            val = log_stats['ema_test_coco_eval_bbox']
            if isinstance(val, (list, tuple)) and len(val) > 0:
                if not self.segmentation_head:
                    ema_map = val[0]
            elif isinstance(val, (float, int)):
                ema_map = val

        if 'ema_test_coco_eval_masks' in log_stats and self.segmentation_head:
            val = log_stats['ema_test_coco_eval_masks']
            if isinstance(val, (list, tuple)) and len(val) > 0:
                ema_map = val[0]
            elif isinstance(val, (float, int)):
                ema_map = val

        # --- Determine the current mAP to compare against best ---
        current_map = None
        metric_source = "unknown"
        
        if regular_map is not None and ema_map is not None:
            if self.use_ema:
                current_map = ema_map
                metric_source = "EMA"
            else:
                current_map = max(regular_map, ema_map)
                metric_source = "max(regular, EMA)"
        elif ema_map is not None:
            current_map = ema_map
            metric_source = "EMA"
        elif regular_map is not None:
            current_map = regular_map
            metric_source = "regular"
        else:
            # No mAP metric was found in this epoch's log
            if self.verbose:
                print("Early stopping: No valid mAP metric found in log_stats.")
            return

        if self.verbose:
            print(f"Early stopping: Current mAP ({metric_source}): {current_map:.4f}, Best: {self.best_map:.4f}, Diff: {current_map - self.best_map:.4f}, Min delta: {self.min_delta}")

        if current_map > self.best_map + self.min_delta:
            self.best_map = current_map
            self.counter = 0
            logger.info(f"Early stopping: mAP improved to {current_map:.4f} using {metric_source} metric")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping: No improvement in mAP for {self.counter} epochs (best: {self.best_map:.4f}, current: {current_map:.4f})")

        if self.counter >= self.patience:
            print(f"Early stopping triggered: No improvement above {self.min_delta} threshold for {self.patience} epochs")
            if self.model and hasattr(self.model, 'stop_training'):
                self.model.stop_training = True
            elif self.model and hasattr(self.model, 'request_early_stop'):
                self.model.request_early_stop()
            else:
                logger.warning("Model does not have stop_training attribute or request_early_stop method.")
