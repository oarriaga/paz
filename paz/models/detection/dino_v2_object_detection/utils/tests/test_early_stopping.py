import os
import sys
import unittest
from unittest.mock import MagicMock

# Dynamic import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import early_stopping

class TestEarlyStopping(unittest.TestCase):
    def test_improvement(self):
        model = MagicMock()
        es = early_stopping.EarlyStoppingCallback(model, patience=2, min_delta=0.1)
        
        # Initial call
        es.update({'test_coco_eval_bbox': [0.5]}) 
        self.assertEqual(es.best_map, 0.5)
        self.assertEqual(es.counter, 0)
        
        # Improvement
        es.update({'test_coco_eval_bbox': [0.7]})
        self.assertEqual(es.best_map, 0.7)
        self.assertEqual(es.counter, 0)
        
    def test_no_improvement(self):
        model = MagicMock()
        es = early_stopping.EarlyStoppingCallback(model, patience=2, min_delta=0.1)
        
        es.update({'test_coco_eval_bbox': [0.5]})
        
        # No improvement (0.55 < 0.5 + 0.1)
        es.update({'test_coco_eval_bbox': [0.55]})
        self.assertEqual(es.counter, 1)
        self.assertEqual(es.best_map, 0.5)
        
        # Still no improvement
        es.update({'test_coco_eval_bbox': [0.58]})
        self.assertEqual(es.counter, 2)
        
        # Should trigger stop
        # Check if request_early_stop was called? 
        # The code calls request_early_stop() if available or sets stop_training
        # Model mock needs these attributes
        
    def test_stop_trigger(self):
        model = MagicMock()
        model.stop_training = False
        es = early_stopping.EarlyStoppingCallback(model, patience=1, min_delta=0.1)
        
        es.update({'test_coco_eval_bbox': [0.5]})
        es.update({'test_coco_eval_bbox': [0.5]}) # Counter = 1, >= patience 1 -> Trigger
        
        # Since MagicMock accepts any attribute set, we check if stop_training was set to True
        # Or if request_early_stop was called
        
        # Our implementation verifies 'stop_training' attr existence first? No, checks `hasattr(model, 'stop_training')`
        # Mocking hasattr on a Mock object is tricky. By default Mock objects return another Mock for attributes.
        # So hasattr(model, 'stop_training') is likely False unless we configure it? 
        # Actually hasattr checks if getattr succeeds. getattr(model, 'stop_training') returns a Mock, so it is "True".
        
        # Let's configure model to have stop_training
        pass # The logic in code: if hasattr(self.model, 'stop_training'): self.model.stop_training = True
        
        # Since we use unittest.mock, we can just assert logic.
        
        # Actually, let's just make a simple class
        class SimpleModel:
            def __init__(self):
                self.stop_training = False
                
        model = SimpleModel()
        es = early_stopping.EarlyStoppingCallback(model, patience=1, min_delta=0.1)
        es.update({'test_coco_eval_bbox': [0.5]})
        es.update({'test_coco_eval_bbox': [0.5]})
        
        assert model.stop_training == True
