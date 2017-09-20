# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***********************************************  Init ****************************************************************
import logging

from PsyNeuLink.Components.Functions.Function import BackPropagation, Exponential, FunctionOutputType, Integrator, \
    Linear, LinearCombination, LinearMatrix, Logistic, SoftMax, UserDefinedFunction
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.System import System, system
from PsyNeuLink.Globals.Keywords import ALL, DEFAULT_MATRIX, EXPONENTS, FULL_CONNECTIVITY_MATRIX, FUNCTION, \
    FUNCTION_PARAMS, HOLLOW_MATRIX, IDENTITY_MATRIX, INITIALIZER, INPUT_STATES, MATRIX, \
    MAX_INDICATOR, MAX_VAL, MONITOR_FOR_CONTROL, OFFSET, OPERATION, OUTPUT_STATES, PARAMETER_STATES, PROB, SCALE, \
    WEIGHTS
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.IntegratorMechanisms import DDM
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms.ComparatorMechanism import \
    ComparatorMechanism
from PsyNeuLink.Scheduling.TimeScale import CentralClock


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
        'PsyNeuLink.Scheduling.Scheduler',
        'PsyNeuLink.Scheduling.Condition',
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
