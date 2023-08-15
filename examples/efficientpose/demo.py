from efficientpose import EFFICIENTPOSEA
import pickle


with open('input_data/input_list.pkl', 'rb') as f:
    input_list = pickle.load(f)


model = EFFICIENTPOSEA()

# Load EfficientNet weights
with open('weights/4.upto_rotationnet_weights_without_iterative_rotationnet.pkl', 'rb') as f:
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

model.set_weights(weights)
detections = model.predict(input_list[0])
print("HI")
