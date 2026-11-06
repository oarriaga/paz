import json
import os
import glob
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# MetricsTracker
# ---------------------------------------------------------------------------


class MetricsTracker:
    """Stateful metric accumulator with checkpoint + plotting support.

    Parameters
    ----------
    output_dir : str
        Root experiment directory (e.g. ``experiments/test``).
    model_name : str
        Base model name used in checkpoint filenames.
    plot_interval : int
        Generate plots every N epochs (and always at the final epoch).
        Set 1 to plot every epoch.
    """

    # Canonical list of tracked metrics — extend here when adding new ones.
    METRIC_KEYS = [
        "epoch",
        "train_loss",
        "val_loss",
        # Detection metrics — validation
        "val_mAP_50",
        "val_mAP_50_95",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_accuracy",
        "val_num_gt_boxes",
        "val_num_pred_boxes",
        # Detection metrics — training (evaluated in inference mode)
        "train_mAP_50",
        "train_mAP_50_95",
        "train_precision",
        "train_recall",
        "train_f1",
        "train_accuracy",
        "train_num_gt_boxes",
        "train_num_pred_boxes",
        # Optimisation
        "learning_rate",
        "grad_norm",
        "grad_norm_max",
        "train_loss_ce",
        "train_loss_bbox",
        "train_loss_giou",
        "val_loss_ce",
        "val_loss_bbox",
        "val_loss_giou",
        "lr_backbone",
        "lr_decoder",
        "lr_head",
    ]

    def __init__(
        self,
        output_dir: str,
        model_name: str = "rfdetr_small",
        plot_interval: int = 1,
        resume: bool = False,
    ):
        self.output_dir = output_dir
        self.model_name = model_name
        self.plot_interval = plot_interval

        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.plots_dir = os.path.join(output_dir, "plots")
        self.log_path = os.path.join(output_dir, "metrics_log.json")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Per-epoch history
        self.history: Dict[str, List[float]] = {
            k: [] for k in self.METRIC_KEYS
        }

        # Per-class metrics (stored separately — variable-length arrays)
        self.per_class_history: Dict[str, list] = {
            "per_class_precision": [],
            "per_class_recall": [],
            "per_class_f1": [],
            "per_class_ap50": [],
        }
        self.class_names: Optional[List[str]] = None

        # Load existing log only when explicitly resuming a run.
        # Loading unconditionally caused epoch-count mismatches when the
        # same experiment directory was reused without --resume.
        if resume and os.path.isfile(self.log_path):
            self._load_log()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float = 0.0,
        # Detection metrics — validation
        val_mAP_50: float = 0.0,
        val_mAP_50_95: float = 0.0,
        val_precision: float = 0.0,
        val_recall: float = 0.0,
        val_f1: float = 0.0,
        val_accuracy: float = 0.0,
        val_num_gt_boxes: int = 0,
        val_num_pred_boxes: int = 0,
        # Detection metrics — training (evaluated in inference mode)
        train_mAP_50: float = 0.0,
        train_mAP_50_95: float = 0.0,
        train_precision: float = 0.0,
        train_recall: float = 0.0,
        train_f1: float = 0.0,
        train_accuracy: float = 0.0,
        train_num_gt_boxes: int = 0,
        train_num_pred_boxes: int = 0,
        # Per-class (val only)
        learning_rate: float = 0.0,
        per_class_precision: Optional[np.ndarray] = None,
        per_class_recall: Optional[np.ndarray] = None,
        per_class_f1: Optional[np.ndarray] = None,
        per_class_ap50: Optional[np.ndarray] = None,
        # Optimisation metrics
        grad_norm: float = 0.0,
        grad_norm_max: float = 0.0,
        train_loss_ce: float = 0.0,
        train_loss_bbox: float = 0.0,
        train_loss_giou: float = 0.0,
        val_loss_ce: float = 0.0,
        val_loss_bbox: float = 0.0,
        val_loss_giou: float = 0.0,
        lr_backbone: float = 0.0,
        lr_decoder: float = 0.0,
        lr_head: float = 0.0,
    ):
        """Record one epoch of scalar and per-class metrics.

        Appends values to ``self.history`` and writes the updated
        JSON log to disk.
        """
        self.history["epoch"].append(int(epoch))
        self.history["train_loss"].append(float(train_loss))
        self.history["val_loss"].append(float(val_loss))
        # Val detection metrics
        self.history["val_mAP_50"].append(float(val_mAP_50))
        self.history["val_mAP_50_95"].append(float(val_mAP_50_95))
        self.history["val_precision"].append(float(val_precision))
        self.history["val_recall"].append(float(val_recall))
        self.history["val_f1"].append(float(val_f1))
        self.history["val_accuracy"].append(float(val_accuracy))
        self.history["val_num_gt_boxes"].append(int(val_num_gt_boxes))
        self.history["val_num_pred_boxes"].append(int(val_num_pred_boxes))
        # Train detection metrics
        self.history["train_mAP_50"].append(float(train_mAP_50))
        self.history["train_mAP_50_95"].append(float(train_mAP_50_95))
        self.history["train_precision"].append(float(train_precision))
        self.history["train_recall"].append(float(train_recall))
        self.history["train_f1"].append(float(train_f1))
        self.history["train_accuracy"].append(float(train_accuracy))
        self.history["train_num_gt_boxes"].append(int(train_num_gt_boxes))
        self.history["train_num_pred_boxes"].append(int(train_num_pred_boxes))
        # Optimisation
        self.history["learning_rate"].append(float(learning_rate))
        self.history["grad_norm"].append(float(grad_norm))
        self.history["grad_norm_max"].append(float(grad_norm_max))
        self.history["train_loss_ce"].append(float(train_loss_ce))
        self.history["train_loss_bbox"].append(float(train_loss_bbox))
        self.history["train_loss_giou"].append(float(train_loss_giou))
        self.history["val_loss_ce"].append(float(val_loss_ce))
        self.history["val_loss_bbox"].append(float(val_loss_bbox))
        self.history["val_loss_giou"].append(float(val_loss_giou))
        self.history["lr_backbone"].append(float(lr_backbone))
        self.history["lr_decoder"].append(float(lr_decoder))
        self.history["lr_head"].append(float(lr_head))

        # Per-class metrics
        _to_list = lambda a: a.tolist() if a is not None else []
        self.per_class_history["per_class_precision"].append(
            _to_list(per_class_precision))
        self.per_class_history["per_class_recall"].append(
            _to_list(per_class_recall))
        self.per_class_history["per_class_f1"].append(
            _to_list(per_class_f1))
        self.per_class_history["per_class_ap50"].append(
            _to_list(per_class_ap50))

        self._save_log()

    def should_plot(self, epoch: int, total_epochs: int) -> bool:
        """Return True if plots should be generated this epoch.

        Plots are generated every ``plot_interval`` epochs and always
        on the final epoch.
        """
        if self.plot_interval <= 1:
            return True
        if (epoch + 1) % self.plot_interval == 0:
            return True
        if epoch == total_epochs - 1:
            return True
        return False

    def generate_plots(self):
        """Write all diagnostic metric plots to ``self.plots_dir``.

        Generates loss curves, mAP curves, precision/recall/F1,
        learning-rate schedule, gradient norms, loss components,
        per-class AP bar chart, and a combined dashboard.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[metrics] matplotlib not available — skipping plots")
            return

        epochs = self.history["epoch"]
        if len(epochs) == 0:
            return

        n = len(epochs)

        def _safe(key):
            """Return history[key] padded/truncated to match epochs."""
            vals = self.history.get(key, [])
            if len(vals) == n:
                return vals
            if len(vals) > n:
                return vals[:n]
            return vals + [0.0] * (n - len(vals))

        # ------------------------------------------------------------------
        # 1. Loss curves (Train & Val)
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, _safe("train_loss"),
                "b-o", markersize=3, label="Train Loss")
        if any(v > 0 for v in _safe("val_loss")):
            ax.plot(epochs, _safe("val_loss"),
                    "r-o", markersize=3, label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "loss_curves.png"), dpi=150)
        plt.close(fig)

        # ------------------------------------------------------------------
        # 2. mAP curves — Train vs Val (mAP@50 and mAP@50:95)
        # ------------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        # mAP@50
        ax1.plot(epochs, _safe("train_mAP_50"),
                 "b-o", markersize=3, label="Train mAP@50")
        ax1.plot(epochs, _safe("val_mAP_50"),
                 "r--s", markersize=3, label="Val mAP@50")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("mAP@50")
        ax1.set_ylim(-0.02, 1.02)
        ax1.set_title("mAP@50 — Train vs Val")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        # mAP@50:95
        ax2.plot(epochs, _safe("train_mAP_50_95"),
                 "b-o", markersize=3, label="Train mAP@50:95")
        ax2.plot(epochs, _safe("val_mAP_50_95"),
                 "r--s", markersize=3, label="Val mAP@50:95")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("mAP@50:95")
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_title("mAP@50:95 — Train vs Val")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        fig.suptitle("Mean Average Precision — Train vs Val",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "mAP_curves.png"), dpi=150)
        plt.close(fig)

        # ------------------------------------------------------------------
        # 3. Precision / Recall / F1 — Train vs Val
        # ------------------------------------------------------------------
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        # Precision
        ax1.plot(epochs, _safe("train_precision"),
                 "b-o", markersize=3, label="Train")
        ax1.plot(epochs, _safe("val_precision"),
                 "r--s", markersize=3, label="Val")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Precision")
        ax1.set_ylim(-0.02, 1.02)
        ax1.set_title("Precision")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        # Recall
        ax2.plot(epochs, _safe("train_recall"),
                 "b-o", markersize=3, label="Train")
        ax2.plot(epochs, _safe("val_recall"),
                 "r--s", markersize=3, label="Val")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Recall")
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_title("Recall")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        # F1
        ax3.plot(epochs, _safe("train_f1"),
                 "b-o", markersize=3, label="Train")
        ax3.plot(epochs, _safe("val_f1"),
                 "r--s", markersize=3, label="Val")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("F1")
        ax3.set_ylim(-0.02, 1.02)
        ax3.set_title("F1 Score")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        fig.suptitle("Precision / Recall / F1 — Train vs Val",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "precision_recall_f1.png"),
                    dpi=150)
        plt.close(fig)

        # ------------------------------------------------------------------
        # 4. Learning rate (per-group if available)
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        has_multi_lr = (
            "lr_backbone" in self.history
            and len(self.history.get("lr_backbone", [])) == n
            and any(v > 0 for v in self.history.get("lr_backbone", []))
        )
        if has_multi_lr:
            ax.plot(epochs, _safe("lr_backbone"),
                    "b-.", markersize=2, label="Backbone LR", alpha=0.8)
            ax.plot(epochs, _safe("lr_decoder"),
                    "g-.", markersize=2, label="Decoder LR", alpha=0.8)
            ax.plot(epochs, _safe("lr_head"),
                    "r-o", markersize=3, label="Head LR")
        else:
            ax.plot(epochs, _safe("learning_rate"),
                    "purple", marker=".", markersize=3, label="LR")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule (per group)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "lr_schedule.png"), dpi=150)
        plt.close(fig)

        # ------------------------------------------------------------------
        # 4b. Gradient norm curve
        # ------------------------------------------------------------------
        has_grad = (
            "grad_norm" in self.history
            and len(self.history.get("grad_norm", [])) == n
            and any(v > 0 for v in self.history.get("grad_norm", []))
        )
        if has_grad:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs, _safe("grad_norm"),
                    "darkorange", marker="o", markersize=3,
                    label="Grad Norm (avg)")
            if "grad_norm_max" in self.history:
                ax.plot(epochs, _safe("grad_norm_max"),
                        "red", marker=".", markersize=2, alpha=0.5,
                        label="Grad Norm (max)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Gradient L2 Norm")
            ax.set_title("Gradient Norm per Epoch")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self.plots_dir,
                                     "gradient_norm.png"), dpi=150)
            plt.close(fig)

        # ------------------------------------------------------------------
        # 4c. Individual loss components (train vs val)
        # ------------------------------------------------------------------
        has_loss_comp = (
            "train_loss_ce" in self.history
            and len(self.history.get("train_loss_ce", [])) == n
            and any(v > 0 for v in self.history.get("train_loss_ce", []))
        )
        has_val_loss_comp = (
            "val_loss_ce" in self.history
            and len(self.history.get("val_loss_ce", [])) == n
            and any(v > 0 for v in self.history.get("val_loss_ce", []))
        )
        if has_loss_comp:
            fig, axes_lc = plt.subplots(1, 3, figsize=(18, 5))
            comp_names = ["loss_ce", "loss_bbox", "loss_giou"]
            comp_titles = ["Cross-Entropy Loss", "BBox L1 Loss", "GIoU Loss"]
            for i, (cname, ctitle) in enumerate(
                    zip(comp_names, comp_titles)):
                axes_lc[i].plot(
                    epochs, _safe(f"train_{cname}"),
                    "b-o", markersize=3, label=f"Train {cname}")
                if has_val_loss_comp:
                    axes_lc[i].plot(
                        epochs, _safe(f"val_{cname}"),
                        "r--s", markersize=3, label=f"Val {cname}")
                axes_lc[i].set_xlabel("Epoch")
                axes_lc[i].set_ylabel("Loss")
                axes_lc[i].set_title(ctitle)
                axes_lc[i].legend(fontsize=8)
                axes_lc[i].grid(True, alpha=0.3)
            fig.suptitle("Loss Components — Train vs Val", fontsize=12,
                         fontweight="bold")
            fig.tight_layout()
            fig.savefig(os.path.join(self.plots_dir,
                                     "loss_components.png"), dpi=150)
            plt.close(fig)

        # ------------------------------------------------------------------
        # 5. Per-class AP@50 bar chart (latest epoch only)
        # ------------------------------------------------------------------
        if (self.per_class_history["per_class_ap50"]
                and self.class_names
                and len(self.per_class_history["per_class_ap50"][-1]) > 0):
            latest_ap = np.array(
                self.per_class_history["per_class_ap50"][-1])
            names = self.class_names[: len(latest_ap)]
            sorted_idx = np.argsort(latest_ap)[::-1]

            fig, ax = plt.subplots(
                figsize=(max(10, len(names) * 0.45), 6))
            bars = ax.bar(range(len(names)),
                          latest_ap[sorted_idx], color="steelblue")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(
                [names[i] for i in sorted_idx],
                rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("AP@50")
            ax.set_ylim(0, 1.05)
            ax.set_title(
                f"Per-Class AP@50 — Epoch {epochs[-1]}")
            ax.grid(True, axis="y", alpha=0.3)
            # Value labels on bars
            for bar, val in zip(bars, latest_ap[sorted_idx]):
                if val > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                            f"{val:.2f}", ha="center", va="bottom",
                            fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(self.plots_dir,
                                     "per_class_ap50.png"), dpi=150)
            plt.close(fig)

        # ------------------------------------------------------------------
        # 6. Combined dashboard (4x2 grid — extended)
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(4, 2, figsize=(16, 18))

        # (0,0) Loss
        axes[0, 0].plot(epochs, _safe("train_loss"),
                        "b-o", markersize=2, label="Train")
        if any(v > 0 for v in _safe("val_loss")):
            axes[0, 0].plot(epochs, _safe("val_loss"),
                            "r-o", markersize=2, label="Val")
        axes[0, 0].set_title("Loss")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        # (0,1) mAP — Train vs Val
        axes[0, 1].plot(epochs, _safe("train_mAP_50"),
                        "b-o", markersize=2, label="Train mAP@50")
        axes[0, 1].plot(epochs, _safe("val_mAP_50"),
                        "r--s", markersize=2, label="Val mAP@50")
        axes[0, 1].plot(epochs, _safe("val_mAP_50_95"),
                        "r:^", markersize=2, label="Val mAP@50:95")
        axes[0, 1].set_title("mAP")
        axes[0, 1].set_ylim(-0.02, 1.02)
        axes[0, 1].legend(fontsize=7)
        axes[0, 1].grid(True, alpha=0.3)

        # (1,0) Precision / Recall — Train vs Val
        axes[1, 0].plot(epochs, _safe("train_precision"),
                        "b-o", markersize=2, label="Train Prec")
        axes[1, 0].plot(epochs, _safe("val_precision"),
                        "r--s", markersize=2, label="Val Prec")
        axes[1, 0].plot(epochs, _safe("train_recall"),
                        "b-^", markersize=2, alpha=0.6, label="Train Rec")
        axes[1, 0].plot(epochs, _safe("val_recall"),
                        "r--v", markersize=2, alpha=0.6, label="Val Rec")
        axes[1, 0].set_title("Precision / Recall")
        axes[1, 0].set_ylim(-0.02, 1.02)
        axes[1, 0].legend(fontsize=7, ncol=2)
        axes[1, 0].grid(True, alpha=0.3)

        # (1,1) F1 — Train vs Val
        axes[1, 1].plot(epochs, _safe("train_f1"),
                        "b-o", markersize=2, label="Train F1")
        axes[1, 1].plot(epochs, _safe("val_f1"),
                        "r--s", markersize=2, label="Val F1")
        axes[1, 1].set_title("F1 Score")
        axes[1, 1].set_ylim(-0.02, 1.02)
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        # (2,0) Loss components (train vs val)
        if has_loss_comp:
            axes[2, 0].plot(epochs, _safe("train_loss_ce"),
                            "b-.", markersize=2, label="train_ce")
            axes[2, 0].plot(epochs, _safe("train_loss_bbox"),
                            "r-.", markersize=2, label="train_bbox")
            axes[2, 0].plot(epochs, _safe("train_loss_giou"),
                            "g-.", markersize=2, label="train_giou")
            if has_val_loss_comp:
                axes[2, 0].plot(epochs, _safe("val_loss_ce"),
                                "b--", markersize=2, alpha=0.6,
                                label="val_ce")
                axes[2, 0].plot(epochs, _safe("val_loss_bbox"),
                                "r--", markersize=2, alpha=0.6,
                                label="val_bbox")
                axes[2, 0].plot(epochs, _safe("val_loss_giou"),
                                "g--", markersize=2, alpha=0.6,
                                label="val_giou")
            axes[2, 0].set_title("Loss Components (Train vs Val)")
            axes[2, 0].legend(fontsize=7, ncol=2)
        else:
            axes[2, 0].plot(epochs, _safe("val_accuracy"),
                            "orange", marker="o", markersize=2)
            axes[2, 0].set_title("Val Accuracy")
            axes[2, 0].set_ylim(-0.02, 1.02)
        axes[2, 0].grid(True, alpha=0.3)

        # (2,1) Gradient norm
        if has_grad:
            axes[2, 1].plot(epochs, _safe("grad_norm"),
                            "darkorange", marker="o", markersize=2,
                            label="Avg")
            if "grad_norm_max" in self.history:
                axes[2, 1].plot(epochs, _safe("grad_norm_max"),
                                "red", marker=".", markersize=1, alpha=0.5,
                                label="Max")
            axes[2, 1].set_title("Gradient Norm")
            axes[2, 1].legend(fontsize=8)
        else:
            axes[2, 1].set_title("(Gradient Norm N/A)")
        axes[2, 1].grid(True, alpha=0.3)

        # (3,0) Learning Rate (per group)
        if has_multi_lr:
            axes[3, 0].plot(epochs, _safe("lr_backbone"),
                            "b-.", markersize=2, label="Backbone")
            axes[3, 0].plot(epochs, _safe("lr_decoder"),
                            "g-.", markersize=2, label="Decoder")
            axes[3, 0].plot(epochs, _safe("lr_head"),
                            "r-o", markersize=2, label="Head")
            axes[3, 0].legend(fontsize=8)
        else:
            axes[3, 0].plot(epochs, _safe("learning_rate"),
                            "purple", marker=".", markersize=2)
        axes[3, 0].set_title("Learning Rate")
        axes[3, 0].grid(True, alpha=0.3)

        # (3,1) Accuracy — Train vs Val
        axes[3, 1].plot(epochs, _safe("train_accuracy"),
                        "b-o", markersize=2, label="Train")
        axes[3, 1].plot(epochs, _safe("val_accuracy"),
                        "r--s", markersize=2, label="Val")
        axes[3, 1].set_title("Accuracy")
        axes[3, 1].set_ylim(-0.02, 1.02)
        axes[3, 1].legend(fontsize=8)
        axes[3, 1].grid(True, alpha=0.3)

        fig.suptitle(f"{self.model_name} — Training Dashboard",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "dashboard.png"), dpi=150)
        plt.close(fig)

        print(f"[metrics] Plots saved to {self.plots_dir}")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def checkpoint_path(
        self, epoch: int, val_loss: float = 0.0, mAP_50: float = 0.0,
    ) -> str:
        """Build a descriptive checkpoint filename.

        Format: ``<model>_epoch_<E>_val_loss_<VL>_mAP_<M>.weights.h5``.
        """
        name = (
            f"{self.model_name}"
            f"_epoch_{epoch:04d}"
            f"_val_loss_{val_loss:.4f}"
            f"_mAP_{mAP_50:.4f}"
            ".weights.h5"
        )
        return os.path.join(self.checkpoint_dir, name)

    def best_checkpoint_path(self) -> str:
        """Path for the 'best' checkpoint."""
        return os.path.join(
            self.checkpoint_dir, f"{self.model_name}_best.weights.h5")

    def find_latest_checkpoint(self) -> Optional[str]:
        """Scan ``checkpoint_dir`` for the checkpoint with the highest epoch.

        Returns the full path, or ``None`` if no checkpoints exist.
        """
        if not os.path.isdir(self.checkpoint_dir):
            return None
        candidates = []
        for fname in os.listdir(self.checkpoint_dir):
            if fname.startswith(self.model_name) and \
               fname.endswith(".weights.h5"):
                if "_best" in fname:
                    continue
                try:
                    parts = fname.replace(".weights.h5", "").split("_epoch_")
                    epoch_part = parts[1].split("_")[0]
                    epoch_num = int(epoch_part)
                    candidates.append((epoch_num, fname))
                except (IndexError, ValueError):
                    continue
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return os.path.join(self.checkpoint_dir, candidates[0][1])

    def find_previous_best(self) -> Optional[str]:
        """Find the most recent 'best' checkpoint (excluding the canonical
        ``_best.weights.h5`` symlink-style file).

        Returns the full path or ``None``.
        """
        pattern = os.path.join(
            self.checkpoint_dir,
            f"{self.model_name}_best_epoch_*.weights.h5")
        matches = glob.glob(pattern)
        if not matches:
            return None
        # Sort by modification time (most recent first)
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]

    def parse_epoch_from_checkpoint(self, ckpt_path: str) -> int:
        """Extract the epoch number from a checkpoint filename."""
        fname = os.path.basename(ckpt_path)
        try:
            parts = fname.replace(".weights.h5", "").split("_epoch_")
            epoch_part = parts[1].split("_")[0]
            return int(epoch_part)
        except (IndexError, ValueError):
            return 0

    @property
    def last_logged_epoch(self) -> int:
        """Return the last epoch recorded in the history, or -1."""
        if self.history["epoch"]:
            return int(self.history["epoch"][-1])
        return -1

    # ------------------------------------------------------------------
    # Summary formatting
    # ------------------------------------------------------------------

    def format_epoch_summary(self, epoch_idx: int = -1) -> str:
        """Return a multi-line human-readable summary for a given epoch."""
        idx = epoch_idx
        if not self.history["epoch"]:
            return "(no data)"
        ep = self.history["epoch"][idx]
        lines = [
            f"  Epoch {ep}:",
            f"    Train Loss     : {self.history['train_loss'][idx]:.4f}",
            f"    Val Loss       : {self.history['val_loss'][idx]:.4f}",
            "    --- Train Eval ---",
            f"    Train mAP@50   : {self.history['train_mAP_50'][idx]:.4f}",
            f"    Train mAP@50:95: {self.history['train_mAP_50_95'][idx]:.4f}",
            f"    Train Precision: {self.history['train_precision'][idx]:.4f}",
            f"    Train Recall   : {self.history['train_recall'][idx]:.4f}",
            f"    Train F1       : {self.history['train_f1'][idx]:.4f}",
            f"    Train Accuracy : {self.history['train_accuracy'][idx]:.4f}",
            f"    Train GT Boxes : {self.history['train_num_gt_boxes'][idx]}",
            f"    Train Pred Box : {self.history['train_num_pred_boxes'][idx]}",
            "    --- Val Eval ---",
            f"    Val mAP@50     : {self.history['val_mAP_50'][idx]:.4f}",
            f"    Val mAP@50:95  : {self.history['val_mAP_50_95'][idx]:.4f}",
            f"    Val Precision  : {self.history['val_precision'][idx]:.4f}",
            f"    Val Recall     : {self.history['val_recall'][idx]:.4f}",
            f"    Val F1         : {self.history['val_f1'][idx]:.4f}",
            f"    Val Accuracy   : {self.history['val_accuracy'][idx]:.4f}",
            f"    Val GT Boxes   : {self.history['val_num_gt_boxes'][idx]}",
            f"    Val Pred Boxes : {self.history['val_num_pred_boxes'][idx]}",
        ]
        # Per-group LR (if available) or single LR
        _lr_bb = self.history.get("lr_backbone", [])
        if _lr_bb and len(_lr_bb) > abs(idx):
            lines.append(
                f"    LR backbone  : {_lr_bb[idx]:.2e}")
            lines.append(
                f"    LR decoder   : "
                f"{self.history['lr_decoder'][idx]:.2e}")
            lines.append(
                f"    LR head      : "
                f"{self.history['lr_head'][idx]:.2e}")
        else:
            lines.append(
                f"    LR           : "
                f"{self.history['learning_rate'][idx]:.2e}")
        # Gradient norm
        _gn = self.history.get("grad_norm", [])
        if _gn and len(_gn) > abs(idx) and _gn[idx] > 0:
            lines.append(f"    Grad Norm    : {_gn[idx]:.4f}")
        # Loss components (train)
        _tlce = self.history.get("train_loss_ce", [])
        if _tlce and len(_tlce) > abs(idx) and _tlce[idx] > 0:
            lines.append(
                f"    Train loss_ce  : {_tlce[idx]:.4f}")
            lines.append(
                f"    Train loss_bbox: "
                f"{self.history['train_loss_bbox'][idx]:.4f}")
            lines.append(
                f"    Train loss_giou: "
                f"{self.history['train_loss_giou'][idx]:.4f}")
        # Loss components (val)
        _vlce = self.history.get("val_loss_ce", [])
        if _vlce and len(_vlce) > abs(idx) and _vlce[idx] > 0:
            lines.append(
                f"    Val loss_ce    : {_vlce[idx]:.4f}")
            lines.append(
                f"    Val loss_bbox  : "
                f"{self.history['val_loss_bbox'][idx]:.4f}")
            lines.append(
                f"    Val loss_giou  : "
                f"{self.history['val_loss_giou'][idx]:.4f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_log(self):
        data = {**self.history, **self.per_class_history}
        if self.class_names:
            data["class_names"] = self.class_names
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_log(self):
        try:
            with open(self.log_path, "r") as f:
                data = json.load(f)
            for key in self.history:
                if key in data:
                    self.history[key] = data[key]
            for key in self.per_class_history:
                if key in data:
                    self.per_class_history[key] = data[key]
            if "class_names" in data:
                self.class_names = data["class_names"]
            # Pad any metric arrays that are shorter than the epoch list
            # (happens when new metrics were added after the log was created)
            n_epochs = len(self.history.get("epoch", []))
            for key in self.history:
                if key == "epoch":
                    continue
                cur_len = len(self.history[key])
                if cur_len < n_epochs:
                    self.history[key].extend(
                        [0.0] * (n_epochs - cur_len)
                    )
            print(f"[metrics] Resumed log with "
                  f"{len(self.history['epoch'])} epoch(s) "
                  f"from {self.log_path}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[metrics] Warning: could not resume log ({e}), "
                  f"starting fresh")
