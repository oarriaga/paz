import time as pytime
import statistics
import jax


def time(
    function,
    num_calls=20,
    warmup_calls=1,
    asynch_dispatch=True,
    *args,
    **kwargs
):
    """Times function with given arguments, after a number of warm-up calls."""
    if not callable(function):
        raise TypeError("The first argument must be a callable function.")
    if not isinstance(num_calls, int) or num_calls < 0:
        raise ValueError("num_calls must be a non-negative integer.")
    if not isinstance(warmup_calls, int) or warmup_calls < 0:
        raise ValueError("num_warmup_calls must be a non-negative integer.")

    for _ in range(warmup_calls):
        function(*args, **kwargs)

    times = []
    if asynch_dispatch:
        for _ in range(num_calls):
            start_time = pytime.perf_counter()
            values = function(*args, **kwargs)
            jax.block_until_ready(values)
            end_time = pytime.perf_counter()
            times.append(end_time - start_time)
    else:
        for _ in range(num_calls):
            start_time = pytime.perf_counter()
            function(*args, **kwargs)
            end_time = pytime.perf_counter()
            times.append(end_time - start_time)

    mean_time = statistics.mean(times) if times else 0.0

    if len(times) < 2:
        stdv_time = 0.0
    else:
        stdv_time = statistics.stdev(times)

    return mean_time, stdv_time
