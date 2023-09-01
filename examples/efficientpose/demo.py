from detection import EFFICIENTPOSEALINEMOD
from paz.backend.image import load_image, show_image
import pickle


with open('input_data/input_list.pkl', 'rb') as f:
    input_list = pickle.load(f)


# Load EfficientNet weights
# with open('weights/6.until_full_iterative_translation_subnets.pkl', 'rb') as f:
#     weights = pickle.load(f)

with open('weights/7.until_full_iterative_translation_subnets_with_bifpn_bug.pkl', 'rb') as f:
    weights = pickle.load(f)

# Load BiFPN + ClassNet + BoxNet weights
# with open('weights/2.bifpn_classbox_net_dict.pkl', 'rb') as f:
#     bifpn_classbox_net_weights = pickle.load(f)

# weights = {}
# weights.update(efficientnet_weights)
# weights.update(bifpn_classbox_net_weights)

# """
# Load weights by assigning weights to corresponding layers by name.
# This is done due to the differences in sequence of layers of paz
# EfficientDet and official implementation of EfficientPose.
# """

# model_layer_names = [layer.name for layer in model.layers]
# for layer_name, layer_weight in list(efficientnet_weights.items()):
#     if layer_name in model_layer_names:
#         model.layers[model_layer_names.index(layer_name)
#                      ].set_weights(layer_weight)

# model.set_weights(weights)
# detections = model.predict(input_list[0])
# print("HI")

detect = EFFICIENTPOSEALINEMOD(score_thresh=0.90, nms_thresh=0.45)
detect.model.set_weights(weights)
image = load_image("/home/manummk95/Desktop/ybkscht_efficientpose/EfficientPose/Datasets/Linemod_preprocessed/data/02/rgb/0247.png")
detections = detect(image)
show_image(detections['image'])

# model = detect.model
# model_layer_names = [layer.name for layer in model.layers]        
# from tensorflow import keras
# inter_output_model = keras.Model(model.input, model.get_layer(index = model_layer_names.index('boxes')).output)
# inter_output = inter_output_model.predict(input_list[0])
# print("")
