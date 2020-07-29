import tensorflow.keras.layers as KL



class AnchorsLayer(KL.Layer):
   
   def __init__(self, name="anchors", **kwargs):
        super(AnchorsLayer, self).__init__(name=name, **kwargs)
   
   def call(self, anchor):
        return anchor
   
   def get_config(self) :
        config = super(AnchorsLayer, self).get_config()
        return config
