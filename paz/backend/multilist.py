class MultiList(object):
    def __init__(self, num_lists):
        self.lists = [[] for _ in range(num_lists)]

    def append(*args):
        for arg_list, arg in zip(self.lists, args):
            arg_list.append(arg)
