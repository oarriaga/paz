from keras.callbacks import LearningRateScheduler


def EpochScheduler(decay_epochs, decay_rate, verbose=1):
    """Keeps learning rate constant until a specified epoch.

    # Arguments:
        decay_epochs: A list of integers representing the epochs at which
            the learning rate should be reduced.
        decay_rate: A float representing the factor by which to reduce the
            learning rate (e.g., 0.1 for a 10x reduction).

    # Returns:
        A ``LearningRateScheduler`` callback instance.
    """

    def schedule(epoch, learning_rate):
        if epoch in decay_epochs:
            new_learning_rate = learning_rate * decay_rate
        else:
            new_learning_rate = learning_rate
        return new_learning_rate

    return LearningRateScheduler(schedule, verbose=verbose)
