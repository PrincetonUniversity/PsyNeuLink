from .regressioncfa import *
from .compositionrunner import *
from .autodiffcomposition import *
from .emcomposition import *
from .grucomposition import *
from psyneulink.library.compositions.emcomposition.emcomposition import *
from psyneulink.library.compositions.grucomposition.grucomposition import *
__all__ = list(regressioncfa.__all__)
__all__.extend(compositionrunner.__all__)
__all__.extend(autodiffcomposition.__all__)
__all__.extend(emcomposition.__all__)
__all__.extend(grucomposition.__all__)

try:
    import torch
    from torch import nn

    # Some torch releases have silent dependency on a more recent numpy than the one curently required by PNL.
    # This breaks torch numpy bindings, see e.g:  https://github.com/pytorch/pytorch/issues/100690
    torch.tensor([1,2,3]).numpy()

    torch_available = True

except (ImportError, RuntimeError):
    torch_available = False

__all__.append('torch_available')
