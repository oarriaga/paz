"""
Serialization overview for paz.inference
=======================================

PAZ provides a safe, versioned serializer for inference objects:

    paz.inference.save(obj, path, format="paz", overwrite=False)
    paz.inference.load(path, format=None, device=None)

Safe format ("paz")
-------------------
- Stored as a directory with:
  - manifest.json (format + version metadata)
  - payload.json  (object spec, no large arrays)
  - arrays.npz    (numerical arrays)
- Does NOT execute arbitrary code on load.
- Requires distribution functions to be explicitly serializable.

Can I serialize my custom PGM?
------------------------------
Yes, if all components are supported by the safe serializer:

1) Priors:
   - Supported distributions: Normal, Uniform, Beta, Bernoulli, Categorical,
     Independent (with supported base distribution).
   - Supported bijectors: Identity, Shift, Scale, Sigmoid, Softplus, Exp, Chain.

2) Latent/Observable nodes:
   - The distribution_fn must be explicitly registered as serializable.
   - Use the @paz.inference.serialization.serializable decorator.
   - Example: linear_regression_likelihood(X) returns a callable with metadata.

3) PGM:
   - The entire graph can be serialized when all nodes meet the above rules.

If you use custom Python closures for distribution_fn, safe serialization
will raise an error. In that case:
    - Rewrite your distribution_fn using a registered builder.

Registering custom distribution builders
----------------------------------------
Register in a module or directly in your script using the decorator.
Example pattern:

    @paz.inference.serialization.serializable("my_custom_likelihood")
    def my_custom_likelihood(feature):
        def apply(param):
            return tfd.Normal(param * feature, 1.0)
        return apply

Now you can use:
    y = paz.Observable(my_custom_likelihood(feature), name="y")(param)

Example: safe serialization round-trip
--------------------------------------
This example uses a registered likelihood so it can be serialized safely.
"""

import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

import paz

tfd = tfp.distributions
tfb = tfp.bijectors


@paz.inference.serialization.serializable("local_linear_regression_likelihood")
def local_linear_regression_likelihood(X):
    def apply(mean, bias, stdv):
        return tfd.Normal(mean * X + bias, stdv)
    return apply


def build_model():
    x = jp.linspace(0.0, 1.0, 8)
    mean = paz.Prior(tfd.Normal(0.0, 1.0), name="mean")
    bias = paz.Prior(tfd.Normal(0.0, 1.0), name="bias")
    low, high = 0.01, 0.3
    bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
    stdv = paz.Prior(tfd.Uniform(low, high), name="stdv", bijector=bijector)
    y_obs = paz.Observable(local_linear_regression_likelihood(x), name="y")(
        mean, bias, stdv
    )
    return paz.PGM([mean, bias, stdv], [y_obs], "serializable_linear")


def main():
    model = build_model()
    key = jax.random.PRNGKey(0)
    data_key, sample_key = jax.random.split(key)
    data = model.sample(data_key, num_samples=1)

    # Serialize the model.
    paz.inference.save(model, "serialization_example", overwrite=True)

    # Load it back.
    loaded = paz.inference.load("serialization_example")

    # Quick check: likelihood log_prob should match.
    inv_sample = model.sample_inverse(sample_key, num_samples=1)
    log_prob = model.likelihood.log_prob(inv_sample, data, space="inv")
    log_prob_loaded = loaded.likelihood.log_prob(inv_sample, data, space="inv")
    print("log_prob:", float(log_prob))
    print("log_prob (loaded):", float(log_prob_loaded))


if __name__ == "__main__":
    main()
