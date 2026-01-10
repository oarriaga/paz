import os
import pickle
import argparse
from glob import glob

import numpy as np
import pandas as pd
import jax
import jax.numpy as jp
from equinox import tree_deserialise_leaves

from tamayo import SE3
from tamayo.render import Render
from tamayo import merge_shapes
from tamayo.camera import build_rays

from primitives import load, flatten, parse_metadata
from lecun import VGG16
from lecun.vgg16 import preprocess_input as preprocess_vgg
from logger import (write_dictionary, build_directory, print_progress,
                    load_parameters, find_path, make_directory)
from plotter import write_image


def extract_image_features(image, model, layer_arg, preprocess):
    x = preprocess(image)
    for layer in model.layers[:layer_arg]:
        x = layer(x)
    return np.array(x)  # numpy is used to prevent memory exhaustion


def denormalize_image(image):
    return 255.0 * image


def extract_features(images, model, layer, preprocess):
    features = []
    for image in images:
        image = denormalize_image(image)
        featuremaps = extract_image_features(image, model, layer, preprocess)
        features.append(featuremaps)
    return np.array(features)


def process_features(features):
    x = features - np.mean(features, axis=(1, 2), keepdims=True)
    x = x / (np.std(features, axis=(1, 2), keepdims=True) + 1e-5)
    x = x * 0.15
    x = x + 0.50
    return np.clip(x, 0, 1)


def compute_feature_error(true_features, pred_features):
    return np.mean((true_features - pred_features)**2, axis=(2, 3))


def render_dataset(render, shapes, floor, lights):
    images = []
    for shape in shapes:
        scene, mask = merge_shapes(floor, shape)
        image, depth = render(scene, mask, lights)
        image = jp.clip(image, 0, 1)
        images.append(image)
    return np.array(images)  # numpy is used to prevent memory exhaustion


def extract_invariances(layer_to_results, arg_to_layer, top_k=5):
    layer_to_errors, layer_to_features = [], []
    num_layers = len(layer_to_results)
    for layer in layer_to_results:
        featuremap, errors = layer
        layer_to_errors.append(errors)
        layer_to_features.append(featuremap)
    layer_to_errors = np.array(layer_to_errors)
    errors = layer_to_errors.reshape(-1)

    lower_error_args = np.argsort(errors)
    row_args = lower_error_args // (num_layers)
    col_args = lower_error_args - (row_args * num_layers)

    row_args = row_args[:top_k]
    col_args = col_args[:top_k]

    invariances = {}
    for arg, (row_arg, col_arg) in enumerate(zip(row_args, col_args)):
        model_layer = int(arg_to_layer[row_arg])
        feature_arg = int(layer_to_features[row_arg][col_arg])
        invariances[arg] = {'layer': model_layer, 'featuremap': feature_arg}
    return invariances


parser = argparse.ArgumentParser(description='Extract neural invariant maps')
parser.add_argument('--seed', default=777, type=int)
parser.add_argument('--root', default='experiments', type=str)
parser.add_argument('--dataset_path', default='datasets', type=str)
parser.add_argument('--dataset_name', default='PRIMITIVES', type=str)
parser.add_argument('--parameters_wildcard', default='*SCENE-OPTIMIZATION')
parser.add_argument('--parameters_filename', default='parameters.json')
parser.add_argument('--shape_directory', default='optimized_shapes')
parser.add_argument('--scene_filename', default='scene.json')
parser.add_argument('--floor_filename', default='floor.npy')
parser.add_argument('--invariance_filename', default='invariances.json')
parser.add_argument('--weights_path', default='VGG16.eqx')
parser.add_argument('--label', default='INVARIANT-MAPS', type=str)
parser.add_argument('--top_k', default=10, type=int)
parser.add_argument('--num_images', default=50, type=int)
parser.add_argument('--shadows', default=True, type=bool)
parser.add_argument('--model', default='VGG16',
                    choices=['VGG16', 'TINYCONVNEXT'], type=str)
args = parser.parse_args()
RNG = np.random.default_rng(args.seed)
key = jax.random.PRNGKey(args.seed)

label = '-'.join([args.model, args.label])
root = build_directory(os.path.join(args.root, args.dataset_name), label)
write_dictionary(args.__dict__, root, 'parameters.json')

wildcard = os.path.join(args.root, args.dataset_name, args.parameters_wildcard)
optimization_metadata = load_parameters(wildcard, args.parameters_filename)
dataset_path = optimization_metadata['dataset_path']
viewport_factor = optimization_metadata['viewport_factor']
split = optimization_metadata['split']
dataset_metadata = parse_metadata(dataset_path, args.dataset_name)
H, W = dataset_metadata['image_shape']
image_shape = [int(H * viewport_factor), int(W * viewport_factor)]
camera_origin = jp.array(dataset_metadata['camera_origin'])
y_FOV = dataset_metadata['y_FOV']

dataset_path = os.path.join(args.dataset_path, args.dataset_name)
dataset = load(dataset_path, split, image_shape)
true_images, depths, labels = flatten(dataset)


optimization_directory = find_path(wildcard)
shapes_directory = optimization_metadata['shapes_directory']
shapes_directory = os.path.join(find_path(wildcard), shapes_directory)
shapes_wildcard = os.path.join(shapes_directory, '*.pkl')
shapes_filenames = sorted(glob(shapes_wildcard))
shapes = []
for shape_filename in shapes_filenames:
    shape = pickle.load(open(f'{shape_filename}', 'rb'))
    shapes.append(shape)

floor_filename = os.path.join(optimization_directory, 'floor.pkl')
floor = pickle.load(open(floor_filename, 'rb'))

lights_filename = os.path.join(optimization_directory, 'lights.pkl')
lights = pickle.load(open(lights_filename, 'rb'))


camera_origin = jp.array(camera_origin)
camera_target = jp.array([0.0, 0.0, 0.0])
camera_upward = jp.array([0.0, 1.0, 0.0])
camera_pose = SE3.view_transform(camera_origin, camera_target, camera_upward)
ray_origins, ray_directions = build_rays(image_shape, y_FOV, camera_pose)
render = Render(ray_origins, ray_directions, *image_shape, args.shadows)
render = jax.jit(render)
pred_images = render_dataset(render, shapes, floor, lights)

if args.model == 'VGG16':
    layers = [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29]
    preprocess = preprocess_vgg
    model = VGG16(key)

model = tree_deserialise_leaves(args.weights_path, model)


image_args = RNG.integers(0, len(true_images), args.num_images)
layer_to_results, results = [], []
for layer_arg, layer in enumerate(layers):
    print_progress(len(layers), layer_arg)
    true_features = extract_features(true_images, model, layer, preprocess)
    pred_features = extract_features(pred_images, model, layer, preprocess)
    dataset_error = compute_feature_error(true_features, pred_features)
    error_per_featuremap = np.mean(dataset_error, axis=0)
    invariant_indices = np.argsort(error_per_featuremap)[:args.top_k]
    invariant_errors = error_per_featuremap[invariant_indices]
    invariant_true_features = true_features[:, invariant_indices]
    invariant_pred_features = pred_features[:, invariant_indices]
    layer_path = os.path.join(root, f'layer_{layer:02d}')
    make_directory(layer_path)

    errors = invariant_errors.tolist()
    indices = invariant_indices.tolist()
    layer_results = {'errors': errors, 'indices': indices}
    write_dictionary(layer_results, layer_path, 'layer_results.json')
    results.extend([indices, errors])
    layer_to_results.append([indices, errors])

    for arg, image_arg in enumerate(image_args):
        image_path = os.path.join(layer_path, f'{arg:02d}_images')
        make_directory(image_path)
        true_image = true_images[image_arg]
        pred_image = pred_images[image_arg]
        true_image = denormalize_image(true_image).astype(np.uint8)
        pred_image = denormalize_image(pred_image).astype(np.uint8)
        write_image(true_image, image_path, 'true_image.png')
        write_image(pred_image, image_path, 'pred_image.png')

        true_featuremaps = invariant_true_features[image_arg]
        pred_featuremaps = invariant_pred_features[image_arg]
        true_featuremaps = process_features(true_featuremaps)
        pred_featuremaps = process_features(pred_featuremaps)
        iterator = enumerate(zip(true_featuremaps, pred_featuremaps))
        for feature_arg, (true_featuremap, pred_featuremap) in iterator:
            true_filename = f'{feature_arg:02d}_true_featuremap.png'
            pred_filename = f'{feature_arg:02d}_pred_featuremap.png'
            write_image(true_featuremap, image_path, true_filename)
            write_image(pred_featuremap, image_path, pred_filename)


invariances = extract_invariances(layer_to_results, layers, args.top_k)
write_dictionary(invariances, root, args.invariance_filename)
layer_names = []
label_names = []
for layer in layers:
    layer_name = f'layer_{layer:02d}'
    layer_names.extend([layer_name, layer_name])
    label_names.extend(['featuremap', 'error'])
column_names = zip(layer_names, label_names)
columns = pd.MultiIndex.from_tuples(column_names)
results = np.array(results)
results = np.moveaxis(results, 0, 1)
index = list(range(args.top_k))
data_frame = pd.DataFrame(results, index=index, columns=columns)
for name in layer_names:
    data_frame[name, 'featuremap'] = data_frame[name, 'featuremap'].astype(int)
data_frame.style.to_latex(os.path.join(root, 'results.tex'))
data_frame.to_csv(os.path.join(root, 'results.csv'))
