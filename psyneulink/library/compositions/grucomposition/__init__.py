# from .grucomposition import *
# from .pytorchGRUwrappers import *
# __all__ = list(grucomposition.__all__)
# __all__.extend(pytorchGRUwrappers.__all__)

try:
    import torch
    from . import grucomposition
    from .grucomposition import *
    from .pytorchGRUwrappers import *
    __all__ = list(grucomposition.__all__)
    __all__.extend(pytorchGRUwrappers.__all__)
except:
    pass
