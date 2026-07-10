def log_prob_inverse(distribution, bijector, inverse_values):
    forward_values = bijector(inverse_values)
    event_ndims = len(distribution.event_shape)
    log_prob = distribution.log_prob(forward_values)
    log_det = bijector.forward_log_det_jacobian(inverse_values, event_ndims)
    return log_prob + log_det
