import ctypes

libObject = ctypes.CDLL('../cpp/example/clibrary.so')

libObject.prompt(20)
