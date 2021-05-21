import math
import colorsys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw

import torch
import torchvision.transforms as transforms

from scenes import SingleView


def length(v):
    return math.sqrt(v[0]**2+v[1]**2)


def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm


def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]


def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=math.acos(cosx) # in radians
   return rad*180/math.pi # returns degrees


def py_ang(A, B=(1,0)):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner


def getAfinityCenter(width, height, point, center, radius=7, img_affinity=None):
    """
    Function to create the affinity maps,
    e.g., vector maps pointing toward the object center.
    Args:
        width: image wight
        height: image height
        point: (x,y)
        center: (x,y)
        radius: pixel radius
        img_affinity: tensor to add to
    return:
        return a tensor
    """
    tensor = torch.zeros(2, height, width).float()

    # Create the canvas for the afinity output
    imgAffinity = Image.new("RGB", (width, height), "black")
    totensor = transforms.Compose([transforms.ToTensor()])

    draw = ImageDraw.Draw(imgAffinity)
    r1 = radius
    p = point
    draw.ellipse((p[0] - r1, p[1] - r1, p[0] + r1, p[1] + r1), (255, 255, 255))

    del draw

    # Compute the array to add the afinity
    array = (np.array(imgAffinity) / 255)[:, :, 0]

    angle_vector = np.array(center) - np.array(point)
    print("Angle vector: {}".format(angle_vector))
    angle_vector = normalize(angle_vector)
    print("Angle vector normalized: {}".format(angle_vector))
    affinity = np.concatenate([[array * angle_vector[0]], [array * angle_vector[1]]])

    # print (tensor)
    if not img_affinity is None:
        # Find the angle vector
        # print (angle_vector)
        if length(angle_vector) > 0:
            angle = py_ang(angle_vector)
        else:
            angle = 0
        # print(angle)
        c = np.array(colorsys.hsv_to_rgb(angle / 360, 1, 1)) * 255
        draw = ImageDraw.Draw(img_affinity)
        draw.ellipse((p[0] - r1, p[1] - r1, p[0] + r1, p[1] + r1), fill=(int(c[0]), int(c[1]), int(c[2])))
        del draw

    re = torch.from_numpy(affinity).float() + tensor
    return re, img_affinity


def GenerateMapAffinity(img, nb_vertex, pointsInterest, objects_centroid, scale):
    """
    Function to create the affinity maps,
    e.g., vector maps pointing toward the object center.
    Args:
        img: PIL image
        nb_vertex: (int) number of points
        pointsInterest: list of points
        objects_centroid: (x,y) centroids for the obects
        scale: (float) by how much you need to scale down the image
    return:
        return a list of tensors for each point except centroid point
    """

    # Apply the downscale right now, so the vectors are correct.
    img_affinity = Image.new(img.mode, (int(img.size[0] / scale), int(img.size[1] / scale)), "black")
    # Create the empty tensors
    totensor = transforms.Compose([transforms.ToTensor()])

    affinities = []
    for i_points in range(nb_vertex):
        affinities.append(torch.zeros(2, int(img.size[1] / scale), int(img.size[0] / scale)))

    for i_pointsImage in range(len(pointsInterest)):
        pointsImage = pointsInterest[i_pointsImage]
        center = objects_centroid[i_pointsImage]
        for i_points in range(nb_vertex):
            point = pointsImage[i_points]
            affinity_pair, img_affinity = getAfinityCenter(int(img.size[0] / scale),
                                                           int(img.size[1] / scale),
                                                           tuple((np.array(pointsImage[i_points]) / scale).tolist()),
                                                           tuple((np.array(center) / scale).tolist()),
                                                           img_affinity=img_affinity, radius=5)

            affinity_pair_np = affinity_pair.cpu().detach().numpy()
            affinities[i_points] = (affinities[i_points] + affinity_pair) / 2

            # Normalizing
            v = affinities[i_points].numpy()

            xvec = v[0]
            yvec = v[1]

            norms = np.sqrt(xvec * xvec + yvec * yvec)
            nonzero = norms > 0

            xvec[nonzero] /= norms[nonzero]
            yvec[nonzero] /= norms[nonzero]

            affinities[i_points] = torch.from_numpy(np.concatenate([[xvec], [yvec]]))
            affinities_np = affinities[i_points].cpu().detach().numpy()
            a = 5

    affinities = torch.cat(affinities, 0)

    return affinities


def CreateBeliefMap(img, pointsBelief, nbpoints, sigma=16):
    """
    Args:
        img: image
        pointsBelief: list of points in the form of
                      [nb object, nb points, 2 (x,y)]
        nbpoints: (int) number of points, DOPE uses 8 points here
        sigma: (int) size of the belief map point
    return:
        return an array of PIL black and white images representing the
        belief maps
    """
    beliefsImg = []
    sigma = int(sigma)
    for numb_point in range(nbpoints):
        array = np.zeros(img.size)
        out = np.zeros(img.size)

        for point in pointsBelief:
            p = point[numb_point]
            w = int(sigma*2)
            if p[0]-w>=0 and p[0]+w<img.size[0] and p[1]-w>=0 and p[1]+w<img.size[1]:
                for i in range(int(p[0])-w, int(p[0])+w):
                    for j in range(int(p[1])-w, int(p[1])+w):
                        array[i,j] = np.exp(-(((i - p[0])**2 + (j - p[1])**2)/(2*(sigma**2))))

        stack = np.stack([array, array, array], axis=0).transpose(2, 1, 0)
        imgBelief = Image.new(img.mode, img.size, "black")
        beliefsImg.append(Image.fromarray((stack*255).astype('uint8')))
    return beliefsImg


file_paths = ["/home/fabian/.keras/datasets/035_power_drill/tsdf/textured.obj"]#, "/home/fabian/.keras/datasets/011_banana/tsdf/textured.obj"]
colors = [np.array([255, 0, 0]), np.array([0, 255, 0])]
view = SingleView(filepath=file_paths, colors=colors, viewport_size=(224, 224))

image_original, alpha_original, bounding_box_points = view.render()

print(bounding_box_points)
pil_images = CreateBeliefMap(Image.fromarray(image_original), bounding_box_points, nbpoints=8, sigma=4)

for pil_image in pil_images:
    pil_image.show()

affinity = GenerateMapAffinity(Image.fromarray(image_original), 8, bounding_box_points, [object_centers[0][:2]], 1).cpu().detach().numpy()
affinity = np.reshape(affinity, (16, 224, 224))
plt.imshow(affinity[0]+affinity[2]+affinity[4]+affinity[6]+affinity[8]+affinity[10]+affinity[12]+affinity[14])
plt.colorbar()
plt.show()

plt.imshow(affinity[1]+affinity[3]+affinity[5]+affinity[7]+affinity[9]+affinity[11]+affinity[13]+affinity[15])
plt.colorbar()
plt.show()
#plt.imshow(affinity[1])
#plt.show()