# Princeton University licenses this file to You under the Apache License,
# Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may
# obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# ***********************************************  Init
# ****************************************************************
import logging

from psyneulink.components.functions.Function import AGTUtilityIntegrator, \
    BackPropagation, Exponential, FHNIntegrator, FunctionOutputType, \
    Integrator, Linear, LinearCombination, LinearMatrix, Logistic, SoftMax, \
    UserDefinedFunction
from psyneulink.components.mechanisms.ProcessingMechanisms.TransferMechanism \
    import TransferMechanism
from psyneulink.components.mechanisms.ProcessingMechanisms \
    .IntegratorMechanism import IntegratorMechanism
from psyneulink.components.Process import process
from psyneulink.components.projections.ModulatoryProjections \
    .ControlProjection import ControlProjection
from psyneulink.components.projections.ModulatoryProjections \
    .LearningProjection import LearningProjection
from psyneulink.components.projections.PathwayProjections.MappingProjection \
    import MappingProjection

from psyneulink.components.ShellClasses import System
from psyneulink.components.System import system
from psyneulink.globals.Keywords import FUNCTION, FUNCTION_PARAMS, \
    INPUT_STATES, PARAMETER_STATES, OUTPUT_STATES, MONITOR_FOR_CONTROL, \
    INITIALIZER, WEIGHTS, EXPONENTS, OPERATION, OFFSET, SCALE, MATRIX, \
    IDENTITY_MATRIX, HOLLOW_MATRIX, FULL_CONNECTIVITY_MATRIX, DEFAULT_MATRIX, \
    ALL, MAX_VAL, MAX_INDICATOR, PROB
from psyneulink.library.mechanisms.ProcessingMechanisms.IntegratorMechanisms \
    import DDM
from psyneulink.library.mechanisms.ProcessingMechanisms.ObjectiveMechanisms \
    .ComparatorMechanism import ComparatorMechanism
from psyneulink.library.subsystems.evc import EVCControlMechanism
from psyneulink.scheduling.TimeScale import CentralClock


# https://stackoverflow.com/a/17276457/3131666
class Whitelist(logging.Filter):
    def __init__(self, *whitelist):
        self.whitelist = [logging.Filter(name) for name in whitelist]

    def filter(self, record):
        return any(f.filter(record) for f in self.whitelist)


class Blacklist(Whitelist):
    def filter(self, record):
        return not Whitelist.filter(self, record)


logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
for handler in logging.root.handlers:
    handler.addFilter(Blacklist(
            'psyneulink.scheduling.Scheduler',
            'psyneulink.scheduling.Condition',
    ))

__all__ = ['System',
           'system',
           'process',
           'CentralClock',
           'TransferMechanism',
           'IntegratorMechanism',
           'DDM',
           'EVCControlMechanism',
           'ComparatorMechanism',
           'MappingProjection',
           'ControlProjection',
           'LearningProjection',
           'UserDefinedFunction',
           'LinearCombination',
           'Linear',
           'Exponential',
           'Logistic',
           'SoftMax',
           'Integrator',
           'LinearMatrix',
           'AGTUtilityIntegrator',
           'FHNIntegrator',
           'BackPropagation',
           'FunctionOutputType',
           'FUNCTION',
           'FUNCTION_PARAMS',
           'INPUT_STATES',
           'PARAMETER_STATES',
           'OUTPUT_STATES',
           'MONITOR_FOR_CONTROL',
           'INITIALIZER',
           'WEIGHTS',
           'EXPONENTS',
           'OPERATION',
           'OFFSET',
           'SCALE',
           'MATRIX',
           'IDENTITY_MATRIX',
           'HOLLOW_MATRIX',
           'FULL_CONNECTIVITY_MATRIX',
           'DEFAULT_MATRIX',
           'ALL',
           'MAX_VAL',
           'MAX_INDICATOR',
           'PROB']
