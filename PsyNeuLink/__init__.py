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

from PsyNeuLink.Components.Functions.Function import *
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism import IntegratorMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ControlProjection import ControlProjection
from PsyNeuLink.Components.Projections.LearningProjection import LearningProjection
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
from PsyNeuLink.Components.System import System, system
from PsyNeuLink.Globals.Defaults import DefaultControlAllocationMode
from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import ComponentPreferenceSet

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__all__ = ['System',
           'system',
           'process',
           'CentralClock',
           'TransferMechanism',
           'IntegratorMechanism',
           'DDM',
           'EVCMechanism',
           'ComparatorMechanism',
           'WeightedErrorMechanism',
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
           'BackPropagation',
           'FunctionOutputType',
           'FUNCTION',
           'FUNCTION_PARAMS',
           'INPUT_STATES',
           'PARAMETER_STATES',
           'OUTPUT_STATES',
           'MAKE_DEFAULT_CONTROLLER',
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
