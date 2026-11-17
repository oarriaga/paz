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
   - Use the @paz.inference.serialize.serializable decorator.
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

    @paz.inference.serialize.serializable("my_custom_likelihood")
    def my_custom_likelihood(feature):
        def apply(param):
            return paz.distributions.Normal(param * feature, 1.0)
        return apply

Now you can use:
    y = paz.Observable(my_custom_likelihood(feature), name="y")(param)

Example: safe serialization round-trip
--------------------------------------
This example uses a registered likelihood so it can be serialized safely.
"""

import argparse

import jax
import jax.numpy as jp

import paz


@paz.inference.serialize.serializable("local_linear_regression_likelihood")
def local_linear_regression_likelihood(X):
    def apply(mean, bias, stdv):
        return paz.distributions.Normal(mean * X + bias, stdv)

    return apply


def build_model(x, low, high):
    standard = paz.distributions.Normal(0.0, 1.0)
    mean = paz.Prior(standard, name="mean")
    bias = paz.Prior(standard, name="bias")
    affine = [paz.bijectors.Shift(low), paz.bijectors.Scale(high - low)]
    bijector = paz.bijectors.Chain(affine + [paz.bijectors.Sigmoid()])
    uniform = paz.distributions.Uniform(low, high)
    stdv = paz.Prior(uniform, bijector, "stdv")
    likelihood = local_linear_regression_likelihood(x)
    y = paz.Observable(likelihood, name="y")(mean, bias, stdv)
    return paz.PGM([mean, bias, stdv], [y], "serializable_linear")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save and load a PGM")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = build_model(jp.linspace(0.0, 1.0, 8), 0.01, 0.3)
    data_key, sample_key = jax.random.split(jax.random.PRNGKey(args.seed))
    data = model.sample(data_key, num_samples=1)

    paz.inference.save(model, "linear_model", overwrite=True)
    loaded = paz.inference.load("linear_model")

    # The loaded model must score the same sample with the same log_prob.
    sample = model.sample_inverse(sample_key, num_samples=1)
    log_prob = model.likelihood.log_prob_inverse(sample, data)
    loaded_log_prob = loaded.likelihood.log_prob_inverse(sample, data)
    print("log_prob:", float(log_prob.log_prob_sum))
    print("log_prob (loaded):", float(loaded_log_prob.log_prob_sum))
