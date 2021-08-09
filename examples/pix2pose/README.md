# Multiple Hypotheses
 
 Watch out for multiple things to get this running:
 
 1. TensorFlow version < 2.4 is required. For TensorFlow 2.4 and above [layer].outputs returns a KerasTensor, but we
    need a TensorFlow Tensor
 2. Somehow we need to set tf.config.experimental_run_functions_eagerly(True) before training