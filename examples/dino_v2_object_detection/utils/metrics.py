from typing import Any, Dict, List, Optional, Sequence, TypeVar
import matplotlib.pyplot as plt
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import wandb
except ImportError:
    wandb = None

plt.ioff()

PLOT_FILE_NAME = "metrics_plot.png"

_T = TypeVar("_T")


def safe_index(arr, idx):
    return arr[idx] if 0 <= idx < len(arr) else None


class MetricsPlotSink:
    """
    The MetricsPlotSink class records training metrics and saves them to a plot.

    Args:
        output_dir (str): Directory where the plot will be saved.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.history = []

    def update(self, values):
        self.history.append(values)

    def save(self):
        if not self.history:
            print("No data to plot.")
            return

        def get_array(key):
            return np.array([h[key] for h in self.history if key in h])

        epochs = get_array('epoch')
        train_loss = get_array('train_loss')
        test_loss = get_array('test_loss')
        
        # Accessing coco_eval safely
        test_coco_eval = [h['test_coco_eval_bbox'] for h in self.history if 'test_coco_eval_bbox' in h]
        ap50_90 = np.array([safe_index(x, 0) for x in test_coco_eval if x is not None], dtype=np.float32)
        ap50 = np.array([safe_index(x, 1) for x in test_coco_eval if x is not None], dtype=np.float32)
        ar50_90 = np.array([safe_index(x, 8) for x in test_coco_eval if x is not None], dtype=np.float32)

        ema_coco_eval = [h['ema_test_coco_eval_bbox'] for h in self.history if 'ema_test_coco_eval_bbox' in h]
        ema_ap50_90 = np.array([safe_index(x, 0) for x in ema_coco_eval if x is not None], dtype=np.float32)
        ema_ap50 = np.array([safe_index(x, 1) for x in ema_coco_eval if x is not None], dtype=np.float32)
        ema_ar50_90 = np.array([safe_index(x, 8) for x in ema_coco_eval if x is not None], dtype=np.float32)

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Subplot (0,0): Training and Validation Loss
        if len(epochs) > 0:
            if len(train_loss):
                axes[0][0].plot(epochs, train_loss, label='Training Loss', marker='o', linestyle='-')
            if len(test_loss):
                axes[0][0].plot(epochs, test_loss, label='Validation Loss', marker='o', linestyle='--')
            axes[0][0].set_title('Training and Validation Loss')
            axes[0][0].set_xlabel('Epoch Number')
            axes[0][0].set_ylabel('Loss Value')
            axes[0][0].legend()
            axes[0][0].grid(True)

        # Subplot (0,1): Average Precision @0.50
        if ap50.size > 0 or ema_ap50.size > 0:
            if ap50.size > 0:
                axes[0][1].plot(epochs[:len(ap50)], ap50, marker='o', linestyle='-', label='Base Model')
            if ema_ap50.size > 0:
                axes[0][1].plot(epochs[:len(ema_ap50)], ema_ap50, marker='o', linestyle='--', label='EMA Model')
            axes[0][1].set_title('Average Precision @0.50')
            axes[0][1].set_xlabel('Epoch Number')
            axes[0][1].set_ylabel('AP50')
            axes[0][1].legend()
            axes[0][1].grid(True)

        # Subplot (1,0): Average Precision @0.50:0.95
        if ap50_90.size > 0 or ema_ap50_90.size > 0:
            if ap50_90.size > 0:
                axes[1][0].plot(epochs[:len(ap50_90)], ap50_90, marker='o', linestyle='-', label='Base Model')
            if ema_ap50_90.size > 0:
                axes[1][0].plot(epochs[:len(ema_ap50_90)], ema_ap50_90, marker='o', linestyle='--', label='EMA Model')
            axes[1][0].set_title('Average Precision @0.50:0.95')
            axes[1][0].set_xlabel('Epoch Number')
            axes[1][0].set_ylabel('AP')
            axes[1][0].legend()
            axes[1][0].grid(True)

        # Subplot (1,1): Average Recall @0.50:0.95
        if ar50_90.size > 0 or ema_ar50_90.size > 0:
            if ar50_90.size > 0:
                axes[1][1].plot(epochs[:len(ar50_90)], ar50_90, marker='o', linestyle='-', label='Base Model')
            if ema_ar50_90.size > 0:
                axes[1][1].plot(epochs[:len(ema_ar50_90)], ema_ar50_90, marker='o', linestyle='--', label='EMA Model')
            axes[1][1].set_title('Average Recall @0.50:0.95')
            axes[1][1].set_xlabel('Epoch Number')
            axes[1][1].set_ylabel('AR')
            axes[1][1].legend()
            axes[1][1].grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{PLOT_FILE_NAME}")
        plt.close(fig)
        print(f"Results saved to {self.output_dir}/{PLOT_FILE_NAME}")


class MetricsTensorBoardSink:
    """
    Training metrics via TensorBoard. (Optional)
    """

    def __init__(self, output_dir):
        if SummaryWriter:
            self.writer = SummaryWriter(log_dir=output_dir)
            print(f"TensorBoard logging initialized.")
        else:
            self.writer = None
            print("Unable to initialize TensorBoard. Logging is turned off.")

    def update(self, values):
        if not self.writer:
            return

        epoch = values.get('epoch', 0)

        if 'train_loss' in values:
            self.writer.add_scalar("Loss/Train", values['train_loss'], epoch)
        if 'test_loss' in values:
            self.writer.add_scalar("Loss/Test", values['test_loss'], epoch)

        # Basic COCO metrics logging
        if 'test_coco_eval_bbox' in values:
            coco_eval = values['test_coco_eval_bbox']
            if safe_index(coco_eval, 0) is not None:
                self.writer.add_scalar("Metrics/Base/AP50_90", coco_eval[0], epoch)

        self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()

class MetricsWandBSink:
    """
    Training metrics via W&B. (Optional)
    """

    def __init__(self, output_dir, project=None, run=None, config=None):
        self.output_dir = output_dir
        if wandb:
            self.run = wandb.init(
                project=project,
                name=run,
                config=config,
                dir=output_dir
            )
            print(f"W&B logging initialized.")
        else:
            self.run = None
            print("Unable to initialize W&B. Logging is turned off.")

    def update(self, values):
        if not wandb or not self.run:
            return
        
        wandb.log(values)

    def close(self):
        if wandb and self.run:
            self.run.finish()
