from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def MLP_A(num_classes=1, hidden_neurons=128):
    model = Sequential(
        [Dense(hidden_neurons, activation='relu', input_shape=(2,)),
         Dense(1, activation='sigmoid')], name='MLP' + str(hidden_neurons))
    return model
