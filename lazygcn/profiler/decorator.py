import functools
import time

from .. import profiler

# TODO: either have @profile.time() decorator or single profiler for all with config
# ! is parrents method decorator called on child overrided method?
# http://witkowskibartosz.com/blog/python_decorators_vs_inheritance.html

def profile(*args):
    print('Vaz', profiler.config.is_enabled)
    if len(args) == 1 and callable(args[0]):
        func = args[0]
        @functools.wraps(func)
        def timed_function(*args, **kw):
            start = time.time()
            result = func(*args, **kw)
            end = time.time()
            print(end-start)
            return result
        return timed_function
    else:
        label = args[]
        enabled = args[1]
        def called(func):
            if profiler.config.is_enabled:
                @functools.wraps(func)
                def timed_function(*args, **kw):
                    start = time.perf_counter()
                    result = func(*args, **kw)
                    end = time.perf_counter()
                    profiler.config.timeing[label].append((end-start))
                    return result
                return timed_function
            else:
                return func
        return called
