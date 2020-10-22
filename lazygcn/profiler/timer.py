import time

class Timer(object):
    def __init__(self, label):
        self.label = label
        self.elapsed = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

# TODO: timing history object