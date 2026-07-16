import jax
import numpy as np

from paz.models.foundation.florence2 import preprocessing
from paz.models.foundation.florence2 import tokenizer as tokenizers
from paz.models.foundation.flower.pretrained import FLOWERLiberoObject
from paz.models.foundation.flower.sampling import sample_actions
from paz.models.foundation.flower.sampling import sample_noise


def PredictFlowerActions(models=None, weights="pretrained", models_path=None,
                         num_flow_steps=None, seed=0):
    if models is None:
        models = FLOWERLiberoObject(weights=weights, models_path=models_path)
    if num_flow_steps is None:
        num_flow_steps = models.config.num_sampling_steps
    key = jax.random.PRNGKey(seed)

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
        inputs = [static, wrist, tokenize(instruction)]
        context = models.encoder.predict(inputs, verbose=0)
        noise = sample_noise(noise_key, 1, models.config)
        actions = sample_actions(models.dit, context, noise, num_flow_steps)
        return np.asarray(actions[0])

    return predict
