from psyneulink.core.components.functions.function import Function_Base
from psyneulink.core.globals.keywords import INTEGRATOR_FUNCTION_TYPE

__all__ = ['IntegratorFunction']

class IntegratorFunction(Function_Base):
    componentType = INTEGRATOR_FUNCTION_TYPE


