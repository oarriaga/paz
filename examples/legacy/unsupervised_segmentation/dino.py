import numpy as np
import torch


def image_to_pytorch_tensor(image):
    image = np.array(image)
    image = torch.from_numpy(np.moveaxis(image, 2, 0))
    image = image.to("cuda")
    return image


def compute_features(model, image):
    image_tensor = image_to_pytorch_tensor(image).unsqueeze(0)
    with torch.inference_mode():
        features = model.forward_features(image_tensor)["x_norm_patchtokens"]
    return features.squeeze().cpu().numpy()
