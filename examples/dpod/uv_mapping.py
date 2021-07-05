import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def create_uv_map(index_first_color=1, index_second_color=2, size=(1024, 1024)):
    """
    Create UV map to wrap the object in it
    :return: uv map (numpy array)
    """
    uv_map_third = np.zeros((size[0], size[1], 1))
    array01 = np.linspace(0, 255, size[0])
    array02 = np.linspace(0, 255, size[1])
    xx, yy = np.meshgrid(array01, array02)
    coords = np.array((xx.ravel(), yy.ravel())).T
    coords = np.reshape(coords, (size[0], size[1], 2))
    uv_map = np.concatenate((uv_map_third, coords), axis=-1)
    uv_map = uv_map.astype(np.uint8)

    im = Image.fromarray(uv_map)
    im.save("uv_map.png")
    plt.imshow(uv_map)
    plt.show()


if __name__ == "__main__":
    create_uv_map(size=(2048, 2048))