import itertools
import numpy as np
import jax.numpy as jp
import jax
import paz

filepath = "phong_specular_shininess"
path = paz.logger.make_directory(filepath)
color = jp.array([0.1, 0.5, 0.8])
ambient = 0.3
diffuse = 0.7
material_args = (color, ambient, diffuse)
light_position = jp.array([3.0, 3.0, 3.0])

H, W = 1028, 1028
image_shape = (H, W)
y_FOV = jp.pi / 2.0
lights = [paz.graphics.PointLight(jp.full(3, 0.9), light_position)]
camera_pose = paz.SE3.view_transform(
    jp.array([0.0, 0.0, 1.5]),
    jp.array([0.0, 0.0, 0.0]),
    jp.array([0.0, 1.0, 0.0]),
)
render_args = image_shape, y_FOV, camera_pose
render_kwargs = dict(lights=lights, tiles=(1, 1), chunk_size=1024)
render = paz.partial(paz.graphics.render, *render_args, **render_kwargs)
render = jax.jit(paz.partial(render, mask=None, shadows=False))


all_shininess = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
all_specular = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
parameters = itertools.product(all_shininess, all_specular)
images = []
for arg, (shininess, specular) in enumerate(parameters):
    material_args_full = (*material_args, specular, shininess, 0.0, 0.0, 1.0)
    material = paz.graphics.Material(*material_args_full)
    scene = paz.graphics.Scene([paz.graphics.Sphere(material=material)])
    image, depth = render(scene=scene)
    image = paz.image.resize(image, (H // 2, W // 2), "bilinear")
    filename = f"{arg}_specular-{specular:.1f}_shininess-{shininess}.png"
    paz.image.write(f"{path}/{filename}", paz.image.denormalize(image))
    images.append(paz.to_numpy(image))

images = paz.to_jax(np.array(images))
mosaic = paz.draw.mosaic(images, (len(all_shininess), len(all_specular)), 0, 1)
paz.image.write("mosaic.png", paz.image.denormalize(mosaic))
