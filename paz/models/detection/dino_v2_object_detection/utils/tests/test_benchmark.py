import os
import sys
import shutil
import tempfile
import pytest
import numpy as np

# Dynamic import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import benchmark

class MockModel:
    def __init__(self):
        # Dummy variable for parameter counting
        self.trainable_variables = [np.ones((10, 10))] 
        
    def __call__(self, inputs):
        # Simulate computation
        import time
        time.sleep(0.001)
        return np.sum(inputs)

def test_benchmark_run():
    # Create temp dir for output
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MockModel()
        
        # Dataset: list of images
        # 10 images of shape (10, 10, 3)
        dataset = [np.random.rand(10, 10, 3).astype(np.float32) for _ in range(10)]
        
        # Run benchmark
        res = benchmark.benchmark(model, dataset, tmpdir)
        
        # Check outputs
        assert "nparam" in res
        assert res["nparam"] == 100
        assert "time" in res
        assert "fps" in res
        
        # Check log file creation
        log_path = os.path.join(tmpdir, "benchmark", "log.txt")
        assert os.path.exists(log_path)
