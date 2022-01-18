from model import DOPE
from ambiguity import MultipleHypotheses

multipleHypotheses = MultipleHypotheses(M=10)

# setting optimizer and compiling model
latent_dimension = 128
model = DOPE(num_stages=6, image_shape=(400, 400, 3), num_belief_maps=9, multipleHypotheses=multipleHypotheses)

model.load_weights("/media/fabian/Data/Masterarbeit/data/models/tless27/dope/multiple_hypotheses/dope_model_epoch_8300_weights.h5")
model.save("/media/fabian/Data/Masterarbeit/data/models/tless27/dope/multiple_hypotheses/dope_model_epoch_8300.h5")