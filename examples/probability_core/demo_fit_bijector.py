import jax

from .bijectors import Chain, Scale, Shift
from .distributions import Normal
from .fitting import fit_bijector


key = jax.random.PRNGKey(7)
source = Normal(0.0, 1.0)
target = Normal(2.0, 0.5)
initial = Chain([Shift(0.0), Scale(1.0)])

fitted, losses = fit_bijector(
    source,
    target,
    initial,
    key,
    num_samples=5000,
    num_steps=1500,
)

draw_key = jax.random.PRNGKey(21)
source_samples = source.sample(4096, seed=draw_key)
fitted_samples = fitted(source_samples)

print("initial bijector:", initial)
print("fitted bijector:", fitted)
print("initial loss:", losses[0])
print("final loss:", losses[-1])
print("target mean/stdv:", 2.0, 0.5)
print(
    "fitted mean/stdv:",
    float(fitted_samples.mean()),
    float(fitted_samples.std()),
)
