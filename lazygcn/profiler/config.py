
from collections import defaultdict

class ProfileConfig(object):
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.timeing = defaultdict(list)
        print('Initiated!!!!!!')
    
    @property
    def is_enabled(self):
        return self.enabled
    
    def disable(self):
        self.enabled = False
    
    def enable(self):
        print('Siktiiiii')
        self.enabled = True

    def flush_stats(self):
        self.timeing = defaultdict(list)

# Global profiler config to use in all profilers
config = ProfileConfig()
