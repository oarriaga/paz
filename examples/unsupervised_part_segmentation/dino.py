import numpy as np
import keras
import keras.ops as ops

def compute_features(model, image):
    # Image is (H, W, 3). Add batch dimension.
    image_tensor = ops.expand_dims(image, axis=0)
    
    # Inference
    # The model call returns a dict if training=False?? 
    # Let's check model code.
    # vision_transformer.py:
    # def call(self, x, masks=None, training=False):
    #     ret = self.forward_features(x, masks=masks, training=training)
    #     if training:
    #         return { ... }
    #     return self.head(ret["x_norm_clstoken"])
    
    # Wait, call() returns only class token output (head output) during inference!
    # "return self.head(ret['x_norm_clstoken'])"
    
    # I need patch tokens.
    # The PyTorch demo used: model.forward_features(image_tensor)["x_norm_patchtokens"]
    
    # The Keras model has a forward_features method.
    features_dict = model.forward_features(image_tensor, training=False)
    features = features_dict["x_norm_patchtokens"]
    
    # features shape: (B, N, D). Squeeze batch.
    features = ops.squeeze(features, axis=0)
    return ops.convert_to_numpy(features)
