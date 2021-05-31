import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

belief_maps = np.load("/media/fabian/Data/Masterarbeit/data/dope/train_data_224px/belief_maps_batch_1.npy")
images = np.load("/media/fabian/Data/Masterarbeit/data/dope/train_data_224px/images_batch_1.npy")

print(images.shape)
print(belief_maps.shape)

index = 100

image = images[index, :, :]
belief_maps = np.sum(belief_maps[index, :, :, :], axis=-1)
belief_maps_resized = np.clip(np.array(Image.fromarray(belief_maps).resize((224, 224))), a_min=0.0, a_max=1.0)
belief_maps_resized = np.expand_dims(belief_maps_resized, axis=-1)

plt.imshow(image + belief_maps_resized)
plt.show()