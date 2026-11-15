
import paz
import paz.utils.plot as plot


normal = paz.distributions.Normal(loc=0.0, scale=1.0)
skewed = paz.distributions.LogNormal(loc=0.0, scale=0.6)

plot.configure(fontsize=12)
figure, axes = plot.subplots(ncols=2, figsize=(12, 4))
plot.discretized_distribution(normal, -3.0, 3.0, 31, axes[0], "N discrete")
plot.discretized_distribution(skewed, 0, 6.0, 25, axes[1], "LogNormal discrete")
figure.tight_layout()
plot.show()
