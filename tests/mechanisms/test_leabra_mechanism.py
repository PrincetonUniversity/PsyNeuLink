import numpy as np
import pytest
from psyneulink.components.mechanisms.processing.transfermechanism import TransferError

from psyneulink.components.component import ComponentError
from psyneulink.components.functions.function import ConstantIntegrator, Exponential, Linear, Logistic, Reduce, \
    Reinforcement, SoftMax
from psyneulink.components.functions.function import ExponentialDist, GammaDist, NormalDist, UniformDist, WaldDist
from psyneulink.components.mechanisms.mechanism import MechanismError
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.states.inputstate import InputStateError
from psyneulink.library.mechanisms.processing.leabramechanism import LeabraMechanism
from psyneulink.globals.utilities import UtilitiesError
from psyneulink.globals.keywords import NAME, MECHANISM, INPUT_STATES, OUTPUT_STATES, PROJECTIONS
from psyneulink.scheduling.timescale import TimeScale

class 