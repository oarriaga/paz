import optax

def LBFGS(learning_rate, memory_size, linesearch):
    return optax.lbfgs(learning_rate, memory_size, True, linesearch)
