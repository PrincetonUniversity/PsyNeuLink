__all__ = []

try:
    import torch
    del torch

    from . import autodiffcomposition
    from . import pytorchmodelcreator
    from . import compositionrunner
    from . import regressioncfa
    from . import regressioncfa

    from .autodiffcomposition import *
    from .pytorchmodelcreator import *
    from .regressioncfa import *
    from .compositionrunner import *

    __all__ = list(autodiffcomposition.__all__)
    __all__.extend(pytorchmodelcreator.__all__)
    __all__.extend(regressioncfa.__all__)
    __all__.extend(compositionrunner.__all__)
except ImportError:
    from . import regressioncfa
    from .regressioncfa import *
    __all__.extend(regressioncfa.__all__)
