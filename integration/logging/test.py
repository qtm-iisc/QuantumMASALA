# import logging
# from qtm.logger import qtmlogger, qtmlogger_set_filehandle

# qtmlogger.setLevel(logging.DEBUG)
# qtmlogger_set_filehandle('test2.log')

# print("Logger Level: ", qtmlogger.logger.level)
# qtmlogger.debug("This is a debug log")
# qtmlogger.info("This is an info log")
# qtmlogger.warning("This is a warning log")
# qtmlogger.error("This is an error log")
# qtmlogger.critical("This is a critical log")

# import time

# @qtmlogger.time
# def sleepy_func():
#     time.sleep(1)
    
    
# sleepy_func()

# @qtmlogger.time
# def buggy_func():
#     raise ValueError("Buggy")

    
# buggy_func()


import functools
import sys
import _testcapi

def bar(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except:
            tp, exc, tb = sys.exc_info()
            _testcapi.set_exc_info(tp, exc, tb)
            del tp, exc, tb
            raise
        return result
    return wrapper

@bar
def f(x):
    return 1 / x

f()