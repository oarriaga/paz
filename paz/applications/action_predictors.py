import jax
import numpy as np

from paz.models.foundation.florence2 import preprocessing
from paz.models.foundation.florence2 import tokenizer as tokenizers
from paz.models.foundation.flower.pretrained import FLOWERLiberoObject
from paz.models.foundation.flower.sampling import sample_actions


def PredictFlowerActions(models=None, weights="pretrained", models_path=None,
                         num_flow_steps=4, seed=0):
    if models is None:
        models = FLOWERLiberoObject(weights=weights, models_path=models_path)
    key = jax.random.PRNGKey(seed)
    horizon = models.config.num_actions
    action_dim = models.config.action_dim

    def tokenize(instruction):
        prompt = tokenizers.build_policy_prompt(instruction)
        token_ids = tokenizers.encode(models.tokenizer, prompt)
        token_ids = [tokenizers.FLOW_TOKEN_ID] + token_ids
        return np.array([token_ids], dtype="int32")

    def predict(static_image, wrist_image, instruction):
        nonlocal key
        key, noise_key = jax.random.split(key)
        static = preprocessing.preprocess(static_image)
        wrist = preprocessing.preprocess(wrist_image)
        token_ids = tokenize(instruction)
        context = models.encoder.predict(
            [static, wrist, token_ids], verbose=0)
        shape = (1, horizon, action_dim)
        noise = jax.random.normal(noise_key, shape, dtype="float32")
        actions = sample_actions(models.dit, context, noise, num_flow_steps)
        return np.asarray(actions[0])

    return predict
