
class Evaluator(object):
    def __init__(self, data):
        self.data = data

    def evaluate(self, model):
        raise NotImplementedError
