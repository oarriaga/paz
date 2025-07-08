import paz
import matplotlib.pyplot as plt

path = "experiments/24-06-2025_11-16-56_SSD300/optimization.log"
log = paz.logger.load_csv(path)
plt.plot(log["epoch"], log["loss"], label="Loss")
plt.plot(log["epoch"], log["val_loss"], label="Loss")
plt.show()
