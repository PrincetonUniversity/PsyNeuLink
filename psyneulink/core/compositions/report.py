# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ************************************************ Report **************************************************************

"""
Reporting is enabled by specifying the reporting arguments of a `Composition`\\'s `execution methods
<Composition_ExecutionMethods>` or the `execute <Mechanism_Base.exeucte>` method of a `Mechanism`.  There are
two types of reporting that can be generated: output reporting and progress reporting.  These can be directed to the
python console or other devices, as described below.

.. _Report_Output:

Output Reporting
----------------

Output reporting provides information about the input and output to a `Mechanism` or to a `Composition` and
its `Nodes <Composition_Nodes>` as they execute.  Options can be specified using a value of `ReportOutput` in
the `reportOutputPref <PreferenceSet_reportOutputPref>` of a Component, or the **report_output** argument of a
Mechanism's `execute <Mechanism_Base.execute>` method or any of a Composition's `execution methods
<Composition_Execution_Methods>`.  If `USE_PREFS <ReportOutput.USE_PREFS>` or `TERSE <ReportOutput.TERSE>` is used,
reporting is generated as execution of each Component occurs;  if `FULL <ReportOutput.FULL>` is used, then the
information is reported at the end of each `TRIAL <TimeScale.TRIAL>` executed.  This always includes the input and
output to a `Mechanism` or a `Composition` and its `Nodes <Composition_Nodes>`, and can also include the values
of their `Parameters`, depending on the specification of the **report_params** argument (using `ReportParams` options`
and/or the `reportOutputPref <PreferenceSet_reportOutputPref>` settings of individual Mechanisms).  The output
for a `nested Composition <Composition_Nested>` is indented relative to the output for the Composition within which
it is nested.  Whether `simulations <OptimizationControlMechanism_Execution>` executed by a Composition's `
<Composition_Controller>` are reported is determined by the **report_simulations** argument, using a
`ReportSimulations` option and, if displayed, is indented relative to the `controller <Composition.controller>`
that executed the simulations.  Output is reported to the devices specified in the **report_to_devices** argument
using the `ReportDevices` options (the Python console by default).

.. _Report_Progress:

Progress Reporting
------------------

Progress reporting provides information about the status of execution of a Composition's `run <Composition.run>`
or `learn <Composition.run>` methods.  It can be enabled/disabled by specifying a `ReportProgress` option in the
**report_progress** argument of either of those methods. If enabled, progress is reported at the end of each `TRIAL
<TimeScale.TRIAL>` of a `Composition`\\'s execution, showing the number of `TRIALS <TimeScale.TRIAL>` that have been
executed and a progress bar. If the number `TRIALS <TimeScale.TRIAL>` to be executed is determinable (e.g.,
the **num_trials** of a Composition's `run <Composition.run>` or `learn <Composition.learn>` method is specified),
estimated time remaining is displayed; if the number of trials is not determinable (e.g., if **inputs** argument is
specified using a generator), then a "spinner" is displayed during execution and the the total number of `TRIALS
<TimeScale.TRIAL>` executed is displayed once complete.  Whether `simulations
<OptimizationControlMechanism_Execution>` executed by an a Composition's `controller <Composition_Controller>` are
included is determined by the **report_simulations** argument using a `ReportSimulations` option.  Progress is
reported to the devices specified in the **report_to_devices** argument using the `ReportDevices` options.

.. technical_note::
    Progress reporting is generated and displayed using a `rich Progress Display
    <https://rich.readthedocs.io/en/stable/progress.html#>`_ object.

.. _Report_Simulations:

Simulations
-----------

Output and progress reporting can include execution in `simulations <OptimizationControlMechanism_Execution>`
of a Composition's `controller <Composition_Controller>`), by specifying a `ReportSimulation` option in the
**report_simulations** argument of a Composition's `run <Composition.run>` or `learn <Composition.run>` methods.

.. _Report_To_Device:

Devices
-------

The device(s) to which reporting is sent can be specified using the **report_to_device** argument of a Mechanism's
`execute <Mechanism_Base.execute>` method or any of a Composition's `execution methods <Composition_Execution_Methods>`;
this can be used to store reports in a Composition's `recorded_reports <Composition.recorded_reports>` attribute;
see `ReportDevices` for options.


COMMENT:
Examples of ReportOutput and ReportParams options, including with more than one trial of execution and with simulations

.. _ReportOutput_Examples:

Examples
--------

Note that the report for the execution of a Composition contains information about the `TRIAL <TimeScale.TRIAL>`
and `TIME_STEP <TimeScale.TIME_STEP>` in which the Mechanism executed.

A more complete report of the execution can be generated using the `Report.FULL` and `Report.USE_PREFS` options in the
**report_output** argument of a Composition's `execution methods <Composition_Execution_Methods>`, that also includes
the input and output for the Composition:

  >>> my_comp = pnl.Composition(pathways=[my_mech])
  >>> my_mech.reportOutputPref = ['integration_rate', 'slope', 'rate']
  >>> my_comp.run(report_output=pnl.ReportOutput.FULL)
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  Composition-0: Trial 0  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃                                                                                                          ┃
  ┃ input: [[0.0]]                                                                                           ┃
  ┃                                                                                                          ┃
  ┃ ┌────────────────────────────────────────────  Time Step 0 ────────────────────────────────────────────┐ ┃
  ┃ │ ╭────────────────────────────────────────── My Mechanism ──────────────────────────────────────────╮ │ ┃
  ┃ │ │ input: 0.0                                                                                       │ │ ┃
  ┃ │ │ ╭──────────────────────────────────────────────────────────────────────────────────────────────╮ │ │ ┃
  ┃ │ │ │ params:                                                                                      │ │ │ ┃
  ┃ │ │ │         integration_rate: 0.5                                                                │ │ │ ┃
  ┃ │ │ │         function: Linear Function-6                                                          │ │ │ ┃
  ┃ │ │ │                 slope: 1.0                                                                   │ │ │ ┃
  ┃ │ │ │         integrator_function: AdaptiveIntegrator Function-1                                   │ │ │ ┃
  ┃ │ │ │                 rate: 0.5                                                                    │ │ │ ┃
  ┃ │ │ ╰──────────────────────────────────────────────────────────────────────────────────────────────╯ │ │ ┃
  ┃ │ │ output: 0.0                                                                                      │ │ ┃
  ┃ │ ╰──────────────────────────────────────────────────────────────────────────────────────────────────╯ │ ┃
  ┃ └──────────────────────────────────────────────────────────────────────────────────────────────────────┘ ┃
  ┃                                                                                                          ┃
  ┃ result: [[0.0]]                                                                                          ┃
  ┃                                                                                                          ┃
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
COMMENT



.. _Report_Options:

Reporting Options
-----------------


"""


import re
import sys
import types
import warnings
from enum import Enum, Flag, auto
from io import StringIO
from typing import Union, Optional

import numpy as np
from rich import print, box
from rich.console import Console, RenderGroup
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import Progress as RichProgress

from psyneulink.core.globals.context import Context
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import FUNCTION_PARAMS, INPUT_PORTS, OUTCOME, OUTPUT_PORTS, VALUE
from psyneulink.core.globals.log import LogCondition
from psyneulink.core.globals.utilities import convert_to_list

__all__ = ['Report', 'ReportOutput', 'ReportParams', 'ReportProgress', 'ReportDevices', 'ReportSimulations',
           'CONSOLE', 'CONTROLLED', 'LOGGED', 'MODULATED', 'MONITORED', 'RECORD', 'DIVERT', 'PNL_VIEW', ]

SIMULATION = 'Simulat'
DEFAULT = 'Execut'
SIMULATIONS = 'simulations'
SIMULATING = 'simulating'
REPORT_REPORT = False # USED FOR DEBUGGING
TRIAL_REPORT = 'trial_report'
RUN_REPORT = 'run_report'
PROGRESS_REPORT = 'progress_report'

# rich console report styles
# node
node_panel_color = 'orange1'
# node_panel_box = box.SIMPLE
node_panel_box = box.ROUNDED
# time_step
time_step_panel_color = 'dodger_blue1'
time_step_panel_box = box.SQUARE
# trial
trial_panel_color = 'dodger_blue3'
trial_input_color = 'green'
trial_output_color = 'red'
trial_panel_box = box.HEAVY
# controller simulaton
controller_panel_color = 'purple'
controller_input_color = 'purple'
controller_output_color = 'blue'
controller_panel_box = box.HEAVY


def _get_sim_number(context):
    try:
        context = context.execution_id
    except (AttributeError, TypeError):
        pass

    try:
        return int(re.search(r'num: (\d+)', context).group(1))
    except (AttributeError, TypeError, ValueError):
        return None


class ReportOutput(Enum):
    """
    Options used in the **report_output** argument of a `Composition`\'s `execution methods
    <Composition_Execution_Methods>` or the `execute <Mechanism_Base.execute>` method of a `Mechanism`, to enable and
    determine the type of output generated by reporting (`Report_Output` for additional information).

    .. technical_note::
        Use of these options is expected in the **report_output** constructor for the `Report` object,
        and are used as the values of its `report_output <Report.report_output>` attribute.

    Attributes
    ----------

    OFF
        suppress output reporting.

    USE_PREFS
        use the `reportOutputPref <PreferenceSet_reportOutputPref>` of each `Composition` and/or `Mechanism` executed
        to determine whether and in what format to report its execution.

    TERSE (aka ON)
        enforce reporting execution of *all* Compositions and/or Mechanisms as they are executed, irrespective of their
        `reportOutputPref <PreferenceSet_reportOutputPref>` settings, using a simple line-by-line format to report each.

    FULL
        enforce formatted reporting execution of *all* Compositions and/or Mechanisms at the end of each
        `TRIAL <TimeScale.TRIAL>` of execution, including the input and output of each, irrespective of their
        `reportOutputPref <PreferenceSet_reportOutputPref>` settings.

        .. technical_note::
            Output is formatted using `rich Panel objects <https://rich.readthedocs.io/en/stable/panel.html>`_.
    """

    OFF = 0
    USE_PREFS = 1
    TERSE = 2
    ON = 2
    FULL = 3


class ReportParams(Enum):
    """
    Options used in the **report_params** argument of a `Composition`\'s `execution methods
    <Composition_Execution_Methods>`, to specify the scope of reporting for values of it `Parameters`
    and those of its `Nodes <Composition_Nodes>` (see `Reporting Parameter values <Report_Params>` under
    `Report_Output` for additional details).

    .. technical_note::
        Use of these options is expected in the **report_output** constructor for the `Report` object,
        and are used as the values of its `report_params <Report.report_params>` attribute.

    Attributes
    ----------

    OFF
        suppress reporting of parameter values.

    USE_PREFS
        defers to `reportOutputPref <PreferenceSet_reportOutputPref>` settings of individual Components.

    MODULATED (aka CONTROLLED)
        report all `Parameters` that are being `modulated <ModulatorySignal.modulation>` (i.e., controlled) by a
        `ControlMechanism` within the `Composition` (that is, those for which the corresponding `ParameterPort`
        receives a `ControlProjection` from a `ControlSignal`.

    CONTROLLED (aka MODULATED)
        report all `Parameters` that are being controlled (i.e., `modulated <ModulatorySignal.modulation>`) by a
        `ControlMechanism` within the `Composition` (that is, those for which the corresponding `ParameterPort`
        receives a `ControlProjection` from a `ControlSignal`).

    MONITORED
        report the `value <Mechanism_Base.value>` of any `Mechanism` that is being `monitored
        <ControlMechanism_Monitor_for_Control>` by a `ControlMechanism` or `ObjectiveMechanism`.

    LOGGED
        report all `Parameters` that are specified to be logged with `LogCondition.EXECUTION`;  see `Log` for
        additional details.

    ALL
        enforce reporting of all `Parameters` of a `Composition` and its `Nodes <Composition_Nodes>`.

    """

    OFF = 0
    MODULATED = auto()
    CONTROLLED = MODULATED
    MONITORED = auto()
    LOGGED = auto()
    ALL = auto()

MODULATED = ReportParams.MODULATED
CONTROLLED = ReportParams.CONTROLLED
MONITORED = ReportParams.MONITORED
LOGGED = ReportParams.LOGGED
ALL = ReportParams.ALL


class ReportProgress(Enum):
    """
    Options used in the **report_progress** argument of a `Composition`\'s `run <Composition.run>` and `learn
    <Composition.learn>` methods, to enable/disable progress reporting during execution of a Composition; see
    `Report_Progress` for additional details (see `Report_Progress` for additional information).


    .. technical_note::
        Use of these options is expected in the **report_progress** constructor for the `Report` object,
        and are used as the values of its `report_progress <Report.report_output>` attribute.

    Attributes
    ----------

    OFF
        suppress progress reporting.

    ON
        enable progress reporting for executions of a Composition.
    """

    OFF = 0
    ON = 1


class ReportSimulations(Enum):
    """
    Options used in the **report_simulations** argument of a `Composition`\'s `run <Composition.run>` and `learn
    <Composition.learn>` methods, to specify whether `simulations <OptimizationControlMechanism_Execution>`
    executed by an a Composition's `controller <Composition_Controller>` are included in output and progress reporting
    (see `Report_Simulations` for additional information).

    .. technical_note::
        Use of these options is expected in the **report_progress** constructor for the `Report` object,
        and are used as the values of its `report_progress <Report.report_output>` attribute.

    Attributes
    ----------

    OFF
        suppress output and progress of simulations.

    ON
        enable output and progress reporting of simulations.
    """

    OFF = 0
    ON = 1


class ReportDevices(Flag):
    """
    Options used in the **report_to_devices** argument of a `Composition`\'s `execution methods
    <Composition_Execution_Methods>` or the `execute <Mechanism_Base.execute>` method of a `Mechanism`, to
    determine the devices to which reporting is directed (see `Report_To_Device` for additional information).

    .. technical_note::
        Use of these options is expected in the **report_to_devices** constructor for the `Report` object,
        and are used as the values of its `_report_to_devices <Report._report_to_devices>` attribute.

    Attributes
    ----------

    CONSOLE
        direct reporting to the console in which PsyNeuLink is running

        .. technical_note::
            output is rendered using the `Console markup <https://rich.readthedocs.io/en/stable/markup.html#>`_
            by a `rich Progress <https://rich.readthedocs.io/en/stable/progress.html>`_ object stored in
            `_instance._rich_progress <Report._rich_progress>`.

    RECORD
        capture reporting in `_recorded_reports <Report._recorded_reports>`; specifying this
        option on its own replaces and suppresses reporting to the console; to continue to generate console
        output, explicitly include `CONSOLE` along with `RECORD` in the argument specification.

    .. technical_note::
        DIVERT
            capture reporting otherwise directed to the rich Console in a UDF-8 formatted string and
            stores it in `_rich_diverted_reports <Report._rich_diverted_reports>`. This option suppresses
            console output and is cumulative (that is, it records the sequences of updates sent to the console
            after each TRIAL) and is intended primarily for unit testing. The `RECORD` option should be used for
            recording output, as it can be used with console output if desired, and reflects the final state of
            the display after execution is complete.

    PNL_VIEW
        direct reporting to the PsyNeuLinkView graphical interface [UNDER DEVELOPMENT].
    """

    CONSOLE = auto()
    RECORD = auto()
    DIVERT = auto()
    PNL_VIEW = auto()

CONSOLE = ReportDevices.CONSOLE
RECORD = ReportDevices.RECORD
DIVERT = ReportDevices.DIVERT
PNL_VIEW = ReportDevices.DIVERT


class ReportError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class OutputReport():
    """
    Object used to package Progress reporting for a call to the `run <Composition.run>` or `learn
    <Composition.learn>` methods of a `Composition`.
    """

    def __init__(self, id, num_trials):
        self.num_trials = num_trials
        self.sim_num = None
        self.rich_task_id = id # used for task id in rich
        self.time_step_report = []
        self.trial_report = []
        self.run_report = []


class Report:
    """
    Provides interface to output and progress reporting.  This is a singleton context object, that returns the
    currently active progress context instance if one has been instantiated already in another scope. It deallocates
    the progress bar when the outermost context is released.

    Arguments
    ---------

    report_output : ReportOutput : default ReportOutput.OFF
        specifies whether to report output of the execution on a trial-by-trial as it is generated;
        see `ReportOutput` for options.

    report_params : list[ReportParams] : default [ReportParams.USE_PREFS]
        specifies which params are reported if ReportOutput.FULL is in effect.

    report_progress : ReportProgress : default ReportProgress.OFF
        specifies whether to report progress of each `TRIAL <TimeScale.TRIAL>` of a `Composition`\\'s execution,
        showing the number of `TRIALS <TimeScale.TRIAL>` that have been executed and a progress bar;  see
        `ReportProgress` for additional details and options.

    report_simulations : ReportSimulations : default ReportSimulations.OFF
        specifies whether to report output and progress for `simulations <OptimizationControlMechanism_Execution>`
        executed by an a Composition's `controller <Composition_Controller>`; see `ReportSimulations` for options.

    report_to_devices : list[ReportDevices] : default [ReportDevices.CONSOLE]
        specifies devices to which output and progress reporting is sent;  see `ReportDevices` for options.

    depth_indent_factor : int : default 4
        specifies amount by which to indent for each level of `nested compositions <Composition_Nested>`
        and/or `simulations <OptimizationControlMechanism_Execution>` for ReportOutput.TERSE.

    padding_indent : int : default 1
        specifies the number of spaces by which to indent the border of each nested Panel
        relative the outer one in which it is nested.

    padding_lines : int  : default 1
        specifies the number of lines below each Panel to separate it from the one below it.

    context : Context : default None

    Attributes
    ----------

    _instance : Report
        singleton instance of class;  contains attributes for:

        * a rich Progress object (`_rich_progress`)
        * a PsyNeuLinkView interface object contained in `_PNL_View` - TBI.

    _reporting_enabled : bool : default False
        identifies whether reporting is enabled;  True if either the **_report_output** or **_report_progress**
        progress arguments of the constructor were specified as not False.

    _report_output : ReportOutput : default ReportOutput.OFF
        determines whether and, if so, what form of output is displayed and/or captured.

    _report_params : list[ReportParams] : default [ReportParams.USE_PREFS]
        determines which params are reported if ReportOutput.FULL is in effect.

    _report_progress : ReportProgress : default ReportProgress.OFF
        determines whether progress is displayed and/or captured.

    _report_simulations : ReportSimulations : default ReportSimulations.OFF
        determines whether reporting occurs for output and/or progress of `simulations
        <OptimizationControlMechanism_Execution>`  carried out by the `controller <Composition_Controller>` of a
        `Composition`.

    _report_to_devices : list[ReportDevices : [ReportDevices.CONSOLE]
        list of devices currently enabled for reporting.

    _use_rich : False, *CONSOLE*, *DIVERT* or list: default *CONSOLE*
        identifies whether reporting to rich is enabled (i.e., if *CONSOLE* and/or *DIVERT* were specified in
        **report_to_devices** argument of constructor.

    _rich_console : bool : default True
        determines whether reporting is sent to _rich_progress console;  True if **CONSOLE** is specified in the
        **report_to_devices** argument of constructor.

    _rich_divert : bool : default True
        determines whether reporting is sent to `_rich_diverted_reports <Report._rich_diverted_reports>;
        True if **DIVERT** is specified in the **report_to_devices** argument of constructor.

    _rich_diverted_reports : str :  default []
        if __rich_divert is True, contains output sent to _rich_progress.console.

    _use_pnl_view : bool : default False
        determines whether reporting is sent to PsyNeuLinkView - TBI.

    _record_reports : bool : default False
        determines whether reporting is recorded in `recorded_reports <Report.recorded_reports>`.

    _recorded_reports : str :  default []
        if _record_reports is True, contains a record of reports generated during execution.

    _recording_enabled : bool : default False
        True if any device is specified other than `CONSOLE <ReportDevices.CONSOLE>`.

    output_reports : dict
        contains entries for each Composition (the key) executed during progress reporting; the value of each
        entry is itself a dict with two entries:
        - one containing OutputReports for executions in DEFAULT_MODE (key: DEFAULT)
        - one containing OutputReports for executions in SIMULATION_MODE (key: SIMULATION)

    _execution_stack : list : default []
        tracks `nested compositions <Composition_Nested>` and `controllers <Composition_Controller>`
        (i.e., being used to `simulate <OptimizationControlMechanism_Execution>` a `Composition`).   Entries
        are the nested Compositions and/or controllers in order of their nesting, and are appended and popped,
        respectively, just before and after they are executed (the former in a Composition's `_execute_controller
        <Composition._execute_controller>` method; and the latter in its `execute <Composition.execute>` method).

    _execution_stack_depth : int : default 0
        depth of nesting of executions, including `nested compositions <Composition_Nested>` and any `controllers
        <Composition_Controller>` currently executing `simulations <OptimizationControlMechanism_Execution>`.

    _nested_comps : bool : default False
        True if there are any `nested compositions <Composition_Nested>`
        in `_execution_stack <Report._execution_stack>`.

    _simulating : bool : default False
        True if there are any `controllers <Composition_Controller>` in `_execution_stack <Report._execution_stack>`
        (that is, current call is nested under a simulation of an outer Composition).

        .. technical_note::
           This is distinct from the state of context.runmode (and used to assign ``run_mode`` in several of the
           methods on Report), which identifies whether the inner-most Composition is currently executing a simulation;
           note that context.runmode is only set to ContextFlags.SIMULATION_MODE once a controller has begun calling
           for simulations, and that it itself is called with context.runmode set to ContextFlags.DEFAULT_MODE; under
           that condition, _simulating may be True, while the ``run_mode`` variable set in a method may be
           SIMULATION.

    _sim_str : str : default ''
        string added to `_trial_header <Report._trial_header>` when context.runmode = ContextFlags.SIMULATION_MODE
        and `_simulating <Report._simulating>` is True.

    _trial_header_stack : str
        header information for `TRIAL <TimeScale.TRIAL>` when report_out=ReportOutput.FULL;  constructed in
        `report_output <Report.report_output>` at the beginning of the trial (when content='trial_init') and pushed
        to the stack; then popped from the stack and used to construct the rich Panel for the `TRIAL <TimeScale.TRIAL>`
        and report it at the end of the `TRIAL <TimeScale.TRIAL>` (when content='trial_end').  This is needed to cache
        the trial_headers across nested executions.  (Note: not needed for ReportOutput.TERSE since trial_header is
        reported as soon as it is constructed, at the beginning of a `TRIAL <TimeScale.TRIAL>`.)

    depth_indent_factor : int : default 2
        determines the amount by which to indent for each level of `nested compositions <Composition_Nested>`
        and/or `simulations <OptimizationControlMechanism_Execution>` for ReportOutput.TERSE.

    padding_indent : int : default 1
        determines the number of spaces by which to indent the border of each nested Panel
        relative the outer one in which it is nested.

    padding_lines : int : default 1
        determines the number of lines below each Panel to separated it from the one below it.

    _ref_count : int : default 0
        tracks how many times object has been referenced;  counter is incremented on each context __enter__
        and decrements on each __exit__, to ensure stop progress is not called until all references have been released.
    """

    _instance = None

    def __new__(cls,
                caller,
                report_output:ReportOutput=ReportOutput.OFF,
                report_params:ReportParams=ReportParams.OFF,
                report_progress:ReportProgress=ReportProgress.OFF,
                report_simulations:ReportSimulations=ReportSimulations.OFF,
                report_to_devices:(list(ReportDevices.__members__), list)=ReportDevices.CONSOLE,
                depth_indent_factor:int = 4,
                padding_indent:int = 1,
                padding_lines:int = 1,
                context:Optional[Context]=None
                ) -> 'Report':

        if cls._instance is None:

            # Validate arguments
            # assert context, "PROGRAM ERROR: Call to Report() without 'context' argument."
            source = f'call to execution method for {caller.name or ""}'
            if not isinstance(report_output, ReportOutput):
                raise ReportError(f"Bad 'report_output' arg in {source}: '{report_output}'; "
                                  f"must be a {ReportOutput} option.")
            if not isinstance(report_progress, ReportProgress):
                raise ReportError(f"Bad 'report_progress' arg in {source}: '{report_progress}'; "
                                  f"must be {ReportProgress} option.")
            if not isinstance(report_simulations, ReportSimulations):
                raise ReportError(f"Bad 'report_simulations' arg in {source}: '{report_simulations}'; "
                                  f"must be {ReportSimulations} option.")
            cls._report_to_devices = convert_to_list(report_to_devices or ReportDevices.CONSOLE)
            if not all(isinstance(a, ReportDevices) for a in cls._report_to_devices):
                raise ReportError(f"Bad 'report_to_devices' arg in {source}: '{report_to_devices}'; "
                                  f"must be a one or a list of {ReportDevices} option(s).")

            # Instantiate instance
            cls._instance = super(Report, cls).__new__(cls)

            # Assign option properties
            cls._report_progress = report_progress
            cls._report_output = report_output
            cls._report_params = convert_to_list(report_params)
            cls._reporting_enabled = report_output is not ReportOutput.OFF or cls._report_progress
            cls._report_simulations = report_simulations
            cls._rich_console = ReportDevices.CONSOLE in cls._report_to_devices
            cls._rich_divert = ReportDevices.DIVERT in cls._report_to_devices
            cls._record_reports = ReportDevices.RECORD in cls._report_to_devices
            cls._recording_enabled = any(i is not ReportDevices.CONSOLE for i in cls._report_to_devices)
            # Enable rich if reporting output or progress and using console or recording
            cls._use_rich = (cls._reporting_enabled
                             and (cls._rich_console or cls._rich_divert or cls._record_reports))
            cls._use_pnl_view = ReportDevices.PNL_VIEW in cls._report_to_devices

            cls._execution_stack = []
            cls._trial_header_stack = []

            cls.depth_indent_factor = depth_indent_factor
            cls.padding_indent = padding_indent
            cls._padding_indent_str = padding_indent * ' '
            cls.padding_lines = padding_lines

            # Instantiate rich progress context object
            # - it is not started until the self.start_output_report() method is called
            # - auto_refresh is disabled to accommodate IDEs (such as PyCharm and Jupyter Notebooks)
            if cls._use_rich:
                # Set up RECORDING
                if cls._record_reports:
                    cls._recording_console = Console()
                # Set up DIVERT
                file = False
                if cls._rich_divert:
                    file = StringIO()
                cls._instance._rich_progress = RichProgress(auto_refresh=False, console=Console(file=file))

            # Instantiate interface to PsyNeuLinkView
            if cls._use_pnl_view:
                warnings.warn("'pnl_view' not yet supported as an option for report_progress of Composition.run()")

            cls.output_reports = {}
            cls._recorded_reports = str()
            cls._rich_diverted_reports = str()

            cls._ref_count = 0

        return cls._instance

    @classmethod
    def _destroy(cls) -> None:
        """
        A simple helper method that deallocates the singleton instance. This is called when we want to fully destroy
        the singleton instance and its member progress counters. This will cause the next call to Report() to
        create a completely new singleton instance.
        """
        cls._instance = None

    def __enter__(self):
        """
        This  returns a singleton of the Report class.
        Returns:
            A new singleton PNL progress context if none is currently active, otherwise, it returns the currently
            active context.
        """

        # If this is the top level call to with Report(), start progress reporting
        if self._ref_count == 0:
            if self._use_rich:
                self._rich_progress.start()

        # Keep track of a reference count of how many times we have given a reference.
        self._ref_count = self._ref_count + 1

        return self

    def __exit__(self, type, value, traceback) -> None:
        """
        Called when the context is closed.
        Args:
            type:
            value:
            traceback:
        Returns:
            Returns None so that exceptions generated within the context are propogated back up
        """

        # We are releasing this reference
        self._ref_count = self._ref_count - 1

        # If all references are released, stop progress reporting and destroy the singleton.
        if self._ref_count == 0:

            # If the rich progress bar is not disabled, stop it.
            if self._use_rich:
                self._rich_progress.stop()

            # Destroy the singleton, very important. If we don't do this, the rich progress
            # bar will grow and grow and never be deallocated until the end of program.
            Report._destroy()

    def start_output_report(self, comp, num_trials, context) -> int:
        """
        Initialize a OutputReport for Composition

        Arguments
        ---------

        comp : Composition

        num_trials : int
            number of trials expected to be executed;  if it is sys.max_size, rich Progress Display is run with an
            `indeterminate progress bar <https://rich.readthedocs.io/en/stable/progress.html#indeterminate-progress>'_.

        context : Context
            context providing information about run_mode (DEFAULT or SIMULATION)

        Returns
        -------

        OutputReport id : int
            id is stored in `output_reports <Report.output_reports>`.
        """

        if not comp:
            assert False, "Report.start_progress() called without a Composition specified in 'comp'."
        if num_trials is None:
            assert False, "Report.start_progress() called with num_trials unspecified."

        # Generate space before beginning of output
        if self._use_rich and not self.output_reports:
            print()

        if comp not in self.output_reports:
            self.output_reports.update({comp:{DEFAULT:[], SIMULATION:[], SIMULATING:False}})

        # Used for accessing progress report and reporting results
        if context.runmode & ContextFlags.SIMULATION_MODE:
            run_mode = SIMULATION
        else:
            run_mode = DEFAULT

        if run_mode is SIMULATION and self._report_simulations is not ReportSimulations.ON:
            return

        if run_mode is SIMULATION:
            # If already simulating, return existing report for those simulations
            if self.output_reports[comp][SIMULATING]:
                return len(self.output_reports[comp][run_mode]) - 1

        if self._use_rich:

            visible = (self._rich_console
                       # progress reporting is ON
                       and self._report_progress is ReportProgress.ON
                       # current run is not a simulation (being run by a controller), or simulation reporting is ON
                       and (not self._simulating or self._report_simulations is ReportSimulations.ON)
                       )

            if comp.verbosePref or REPORT_REPORT:
                from pprint import pprint
                pprint(f'{comp.name} {str(context.runmode)} START')

            # when num_trials is not known (e.g., a generator is for inputs)
            if num_trials == sys.maxsize:
                start = False
                num_trials = 0
            else:
                start = True

            self._depth_indent_i = self._depth_str_i = ''
            if run_mode is SIMULATION or self._execution_stack_depth:
                self._depth_indent_i = self.depth_indent_factor * self._execution_stack_depth * ' '
                self._depth_str_i = f' (depth: {self._execution_stack_depth})'

            id = self._rich_progress.add_task(f"[red]{self._depth_indent_i}{comp.name}: "
                                              f"{run_mode}ing {self._depth_str_i}...",
                                              total=num_trials,
                                              start=start,
                                              visible=visible
                                              )

            self.output_reports[comp][run_mode].append(OutputReport(id, num_trials))
            report_num = len(self.output_reports[comp][run_mode]) - 1

            self.output_reports[comp][SIMULATING] = run_mode is SIMULATION

            return report_num

    def report_progress(self, caller, report_num:int, context:Context):
        """
        Report progress of executions in call to `run <Composition.run>` or `learn <Composition.learn>` method of
        a `Composition`, and record reports if specified.

        Arguments
        ---------

        caller : Composition or Mechanism

        report_num : int
            id of OutputReport for caller[run_mode] in self.output_reports to use for reporting.

        context : Context
            context providing information about run_mode (DEFAULT or SIMULATION).
        """

        if self._report_progress is ReportProgress.OFF:
            return

        simulation_mode = context.runmode & ContextFlags.SIMULATION_MODE
        if simulation_mode:
            run_mode = SIMULATION
            # Return if (nested within) a simulation and not reporting simulations
            if self._report_simulations is ReportSimulations.OFF:
                return
        else:
            run_mode = DEFAULT

        output_report = self.output_reports[caller][run_mode][report_num]
        trial_num = self._rich_progress.tasks[output_report.rich_task_id].completed

        # Useful for debugging:
        if caller.verbosePref or REPORT_REPORT:
            from pprint import pprint
            pprint(f'{caller.name} {str(context.runmode)} REPORT')

        # Not in simulation and have reached num_trials (if specified) (i.e. end of run or set of simulations)
        if self.output_reports[caller][SIMULATING] and not simulation_mode:
            # If was simulating previously, then have just exited, so:
            #   (note: need to use transition and not explicit count of simulations,
            #    since number of simulation trials being run is generally not known)
                # - turn it off
            self.output_reports[caller][SIMULATING] = False

        # Update progress report
        if self._use_rich:
            if output_report.num_trials:
                if simulation_mode:
                    num_trials_str = ''
                else:
                    num_trials_str = f' of {output_report.num_trials}'
            else:
                num_trials_str = ''

            # Construct update text
            self._depth_indent = self._depth_str = ''
            if simulation_mode or self._execution_stack_depth:
                self._depth_indent = self.depth_indent_factor * self._execution_stack_depth * ' '
                self._depth_str = f' (depth: {self._execution_stack_depth})'
            update = f'{self._depth_indent}{caller.name}: ' \
                     f'{run_mode}ed {trial_num+1}{num_trials_str} trials{self._depth_str}'

            # Do update
            self._rich_progress.update(output_report.rich_task_id,
                                  description=update,
                                  advance=1,
                                  refresh=True)

        if self._report_progress is not ReportProgress.OFF:
            self._print_and_record_reports(PROGRESS_REPORT, context)


    def report_output(self,
                      caller,
                      report_num:int,
                      scheduler,
                      report_output:ReportOutput,
                      report_params:list,
                      content:str,
                      context:Context,
                      nodes_to_report:bool=False,
                      node=None):
        """
        Report output of execution in call to `execute <Composition.execute>` method of a `Composition` or a
        Mechanism <Mechanism_Base.execute>`.  Report.TERSE generates a line-by-line report of executions, but
        no other information (ie., no input, output or parameter information); output is generated in every call;
        Report.FULL generates a rich-formatted report, that includes that information;  it is constructed throughout
        the execution of the `TRIAL <TimeScale.TRIAL>` (beginning with content='trial_init'), and reported at the end
        of the `TRIAL <TimeScale.TRIAL>`(content='trial_end').

        Arguments
        ---------

        caller : Composition or Mechanism
            Component requesting report;  used to identify relevant output_report.

        report_num : int
            specifies id of `OutputReport`, stored in `output_reports <Report.output_reports>` for each
            Composition executed and mode of execution (DEFAULT or SIMULATION).

        scheduler : Scheduler
            specifies Composition `Scheduler` used to determine the `TIME_STEP <TimeScale.TIME_STEP>` of the current
            execution.

        report_output : ReportOutput
            conveys `ReportOutput` option specified in the **report_output** argument of the call to a Composition's
            `execution method <Composition_Execution_Method>` or a Mechanism's `execute <Mechanism_Base.execute>`
            method.

        report_params : [ReportParams]
            conveys `ReportParams` option(s) specified in the **report_params** argument of the call to a Composition's
            `execution method <Composition_Execution_Method>` or a Mechanism's `execute <Mechanism_Base.execute>`
            method.

        content : str
            specifies content of current element of report;  must be: 'trial_init', 'time_step_init', 'node',
            'time_step', 'trial_end', 'controller_start', 'controller_end', 'run_start, or 'run_end'.

        context : Context
            context of current execution.

        nodes_to_report : bool : default False
            specifies whether there are any nodes to report in current execution;  used to determine
            whether to generate a heading (if report_output = ReportOutput.TERSE mode) or `rich Panel
            <https://rich.readthedocs.io/en/stable/panel.html>`_ (if report_output = ReportOutput.TERSE mode)
            in the output report.

        node : Composition or Mechanism : default None
            specifies `node <Composition_Nodes>` for which output is being reported.
        """

        # if report_num is None or report_output is ReportOutput.OFF:
        if report_output is ReportOutput.OFF:
            return

        # FIX: ASSIGN SCHEDULE IF IT IS NONE (i.e., FROM MECHANISM):  GET IT FROM LATEST COMP ON STACK

        # Determine report type and relevant parameters ----------------------------------------------------------------

        # Get ReportOutputPref for node and whether it is a controller
        if node:
            node_pref = next((pref for pref in convert_to_list(node.reportOutputPref)
                                         if isinstance(pref, ReportOutput)), None)
            if hasattr(node, 'composition') and node.composition:
                is_controller = True
            else:
                is_controller = False

        # Assign report_output as default for trial_report_type and node_report_type...
        trial_report_type = node_report_type = report_output

        # then try to get them from caller, based on whether it is a Mechanism or Composition
        from psyneulink.core.compositions.composition import Composition
        from psyneulink.core.components.mechanisms.mechanism import Mechanism
        # Report is called for by a Mechanism
        if isinstance(caller, Mechanism):
            if context.source & ContextFlags.COMPOSITION:
                output_report_owner = context.composition
                trial_report_type=report_output
            # FULL output reporting doesn't make sense for a Mechanism, since it includes trial info, so enforce TERSE
            else:
                trial_report_type = None
            # If USE_PREFS is specified by user, then assign output format to Mechanism's reportOutputPref
            if report_output is ReportOutput.USE_PREFS:
                node_report_type = node_pref
                if node_pref is ReportOutput.OFF:
                    return
        elif isinstance(caller, Composition):
            # USE_PREFS is specified for report called by a Composition:
            if report_output is ReportOutput.USE_PREFS:
                # First, if report is for execution of a node, assign its report type using its reportOutputPref:
                if node:
                    # Get ReportOutput spec from reportOutputPref if there is one
                    # If None was found, assign ReportOutput.FULL as default
                    node_report_type = node_pref or ReportOutput.FULL
                    # Return if it is OFF
                    if node_report_type is ReportOutput.OFF:
                        return
            output_report_owner = caller

        if scheduler:
            trial_num = scheduler.get_clock(context).time.trial
        else:
            trial_num = None

        self._sim_str = ''

        # Determine run_mode and assign relevant header info
        # if call is from a Composition or a Mechanism being executed by one
        if isinstance(caller, Composition) or context.source == ContextFlags.COMPOSITION:
            simulation_mode = context.runmode & ContextFlags.SIMULATION_MODE
            if (simulation_mode or self._simulating) and self._report_simulations is ReportSimulations.OFF:
                return

            # Track simulation count within each simulation set:
            if content == 'trial_init':

                # FIX: SHOULDN'T POPULATE sim_str IF self._simulating IS FALSE, BUT IT IS AND NOT DOING SO CRASHES
                # if self._simulating:
                if self.output_reports[caller][SIMULATING]:
                    if not simulation_mode:
                        # If was simulating previously but not now in SIMULATION_MODE, then have just exited,
                        #   so reset sim_num
                        #   (note: need to use transition and not explicit count of simulations,
                        #    since number of simulation trials being run is generally not known)
                        self.output_reports[caller][SIMULATION][report_num].sim_num = None
                        self._sim_str = ''
                    else:
                        if self.output_reports[caller][SIMULATION][report_num].sim_num is None:
                            # This is the first simulation, so set to 0
                            self.output_reports[caller][SIMULATION][report_num].sim_num = 0
                        else:
                            # This is a new simulation, so increment number
                            self.output_reports[caller][SIMULATION][report_num].sim_num += 1
                        sim_num = self.output_reports[caller][SIMULATION][report_num].sim_num
                        self._sim_str = f' SIMULATION {sim_num}'

            if simulation_mode:
                # Actual simulation execution
                run_mode = SIMULATION
            elif self._simulating:
                # Composition or controller executing in simulation happens in DEFAULT_MODE
                run_mode = DEFAULT
            else:
                # Non-simulation (but potentially nested) execution
                run_mode = DEFAULT
            output_report = self.output_reports[output_report_owner][run_mode][report_num]

            # FIX: GENERALIZE THIS, PUT AS ATTRIBUTE ON Report, AND THEN REFERENCE THAT IN report_progress
            depth_indent = 0
            if simulation_mode or self._execution_stack_depth:
                depth_indent = self.depth_indent_factor * self._execution_stack_depth

        # Construct output report -----------------------------------------------------------------------------

        if content == 'run_start':
            if simulation_mode:
                self._execution_stack.append(caller.controller)
            else:
                self._execution_stack.append(caller)

            if trial_report_type is ReportOutput.TERSE and not self._simulating:
                # Report execution at start of run, in accord with TERSE reporting at initiation of execution
                report = f'[bold {trial_panel_color}]{self._depth_indent_i}Execution of {caller.name}:[/]'
                self._rich_progress.console.print(report)
                if self._record_reports:
                    self._recorded_reports += report
            return

        if content == 'trial_init':

            self._execution_stack.append(caller)

            output_report.trial_report = []
            #  if FULL output, report trial number and Composition's input
            #  note:  header for Trial Panel is constructed under 'content is Trial' case below
            if trial_report_type is ReportOutput.FULL:
                output_report.trial_report = [f'[bold {trial_panel_color}]{self._padding_indent_str}input:[/]'
                                           f' {[i.tolist() for i in caller.get_input_values(context)]}']
                # Push trial_header to stack in case there are intervening executions of nested comps or simulations
                self._trial_header_stack.append(
                    f'[bold{trial_panel_color}] {caller.name}{self._sim_str}: Trial {trial_num}[/] ')
            else: # TERSE output
                # FIX: ADD THE FOLLOWING FOR NESTED COMP'S, BEFORE trial_header

                trial_header = ''

                # If nested Composition (indicated by current and previous entries on the stack being Compositions)
                previous_caller = self._execution_stack[-2]
                if (isinstance(caller, Composition) and isinstance(previous_caller, Composition)):
                    trial_header += f'{self._depth_indent_i}[bold {trial_panel_color}]Execution of {caller.name} ' \
                                    f'within {previous_caller.name}:[/]\n'

                # print trial title and separator + input array to Composition
                trial_header += f'[bold {trial_panel_color}]' \
                                f'{depth_indent * " "}{caller.name}{self._sim_str} TRIAL {trial_num} ' \
                                f'===================='
                self._rich_progress.console.print(trial_header)
                if self._record_reports:
                    self._recorded_reports += trial_header

        elif content == 'time_step_init':
            if trial_report_type is ReportOutput.FULL:
                output_report.time_step_report = [] # Contains rich.Panel for each node executed in time_step
            elif nodes_to_report: # TERSE output

                time_step_header = f'[{time_step_panel_color}]' \
                                   f'{depth_indent * " "}Time Step {scheduler.get_clock(context).time.time_step} ' \
                                   f'---------'
                self._rich_progress.console.print(time_step_header)
                if self._record_reports:
                    self._recorded_reports += time_step_header

        elif content in {'node', 'nested_comp', 'controller_start'} :
            if not node:
                assert False, 'Node not specified in call to Report report_output'

            if content == 'nested_comp':
                # Assign last run_report for execution of nested_comp (node) as node_report
                title = f'[bold{trial_panel_color}]EXECUTION OF {node.name}[/] within {caller.name}'
                nested_comp_run_report = Padding.indent(Panel(RenderGroup(*(self.output_reports[node][DEFAULT][
                                                                                -1].run_report)),
                                                              box=trial_panel_box,
                                                              border_style=trial_panel_color,
                                                              title=title,
                                                              padding=self.padding_lines,
                                                              expand=False),
                                                        self.padding_indent)
                node_report = nested_comp_run_report

            else:
                # - controller is assigned a report here for use with ReportOutput.TERSE;
                #              for ReportOutput.FULL, its report is assigned after execution of simulations
                node_report = self.node_execution_report(node,
                                                         input_val=node.get_input_values(context),
                                                         output_val=node.output_port.parameters.value._get(context),
                                                         report_output=node_report_type,
                                                         report_params=report_params,
                                                         trial_num=trial_num,
                                                         is_controller=is_controller,
                                                         context=context
                                                         )
            if trial_report_type is ReportOutput.FULL:
                if content=='start_controller':
                    # skip controller, as its report is assigned after execution of simulations
                    return
                output_report.time_step_report.append(node_report)

            # For ReportOutput.TERSE, just print it to the console
            elif trial_report_type is ReportOutput.TERSE:

                if content == 'nested_comp':
                    # Execution of nested Composition is reported before XXX
                    return

                self._rich_progress.console.print(node_report)
                if self._record_reports:
                    with self._recording_console.capture() as capture:
                        self._recording_console.print(node_report)
                    self._recorded_reports += capture.get()

        elif content == 'time_step':
            if nodes_to_report and trial_report_type is ReportOutput.FULL:
                output_report.trial_report.append('')
                title = f'[bold {time_step_panel_color}]\nTime Step {scheduler.get_clock(context).time.time_step}[/]'
                output_report.trial_report.append(Padding.indent(Panel(RenderGroup(*output_report.time_step_report),
                                                                       # box=box.HEAVY,
                                                                       border_style=time_step_panel_color,
                                                                       box=time_step_panel_box,
                                                                       title=title,
                                                                       padding=self.padding_lines,
                                                                       expand=False),
                                                                 self.padding_indent))

        elif content == 'trial_end':
            if trial_report_type is ReportOutput.FULL:
                output_values = []
                for port in caller.output_CIM.output_ports:
                    output_values.append(port.parameters.value._get(context))
                output_report.trial_report.append(f"\n[bold {trial_output_color}]{self._padding_indent_str}result:[/]"
                                          f" {[r.tolist() for r in output_values]}")
                if self._simulating:
                    # If simulating, get header that was stored at the beginning of the simulation set
                    title = self._trial_header_stack.pop()
                else:
                    title = f'[bold{trial_panel_color}] {caller.name}{self._sim_str}: Trial {trial_num}[/] '
                output_report.trial_report = Padding.indent(Panel(RenderGroup(*output_report.trial_report),
                                                                  box=trial_panel_box,
                                                                  border_style=trial_panel_color,
                                                                  title=title,
                                                                  padding=self.padding_lines,
                                                                  expand=False),
                                                            self.padding_indent)

                # # TEST PRINT:
                # self._rich_progress.console.print(output_report.trial_report)

                output_report.run_report.append('')
                output_report.run_report.append(output_report.trial_report)

            self._execution_stack.pop()

        elif content == 'controller_end':

            # Only deal with ReportOutput.FULL;  ReportOutput.TERSE is handled above under content='controller_start'
            if report_output is ReportOutput.FULL:

                features = [p.parameters.value.get(context).tolist() for p in node.input_ports if p.name != OUTCOME]
                outcome = node.input_ports[OUTCOME].parameters.value.get(context).tolist()
                control_allocation = [r.tolist() for r in node.control_allocation]

                ctlr_report = [f'[bold {controller_input_color}]{self._padding_indent_str}features:[/] {features}'
                               f'\n[bold {controller_input_color}]{self._padding_indent_str}outcome:[/] {outcome}']
                ctlr_report.extend(self.output_reports[output_report_owner][SIMULATION][report_num].run_report)
                # MODIFIED 3/25/21 NEW: FIX: THIS DOESN'T SEEM TO DO ANYTHING
                ctlr_report.extend(self.output_reports[output_report_owner][DEFAULT][report_num].run_report)
                # MODIFIED 3/25/21 END
                ctlr_report.append(f"\n[bold {controller_output_color}]{self._padding_indent_str}control allocation:[/]"
                                   f" {control_allocation}")
                ctlr_report = Padding.indent(Panel(RenderGroup(*ctlr_report),
                                                   box=controller_panel_box,
                                                   border_style=controller_panel_color,
                                                   title=f'[bold{controller_panel_color}] {node.name} ' \
                                                         f'SIMULATION OF {node.composition.name}[/] ',
                                                   padding=self.padding_lines,
                                                   expand=False),
                                             self.padding_indent)
                self.output_reports[caller][DEFAULT][-1].run_report.append('')
                self.output_reports[caller][DEFAULT][-1].run_report.append(ctlr_report)

                # # TEST PRINT:
                # self._rich_progress.console.print(ctlr_report)

        elif content == 'run_end':

            self._execution_stack.pop()

            if trial_report_type is ReportOutput.TERSE:
                # Report handled at beginning of run, in accord with TERSE in which lines indicate order of initiation
                return

            if len(self._execution_stack) == 0 and trial_report_type is not ReportOutput.OFF:
                title = f'[bold{trial_panel_color}]EXECUTION OF {node.name}[/] '
                output_report.run_report = Padding.indent(Panel(RenderGroup(*output_report.run_report),
                                                                box=trial_panel_box,
                                                                border_style=trial_panel_color,
                                                                title=title,
                                                                padding=self.padding_lines,
                                                                expand=False),
                                                          self.padding_indent)
                self._print_and_record_reports(RUN_REPORT, context, output_report)

        else:
            assert False, f"Bad 'content' argument in call to Report.report_output() for {caller.name}: {content}."

        return

    def node_execution_report(self,
                              node,
                              input_val:Optional[np.ndarray]=None,
                              output_val:Optional[np.ndarray]=None,
                              report_output=ReportOutput.USE_PREFS,
                              report_params:ReportParams=ReportParams.OFF,
                              trial_num:Optional[int]=None,
                              is_controller=False,
                              context:Context=None) -> Panel:
        """
        Generates formatted output report for the `node <Composition_Nodes>` of a `Composition` or a `Mechanism`.
        Called by `report_output <Report.report_output>` for execution of a Composition, and directly by the `execute
        <Mechanism_Base>` method of a `Mechanism` when executed on its own.

        Allows user to specify *PARAMS* or 'parameters' to induce reporting of all parameters, and a listing of
        individual parameters to list just those.

        Arguments
        ---------

        node : Composition or Mechanism
            node for which report is being requested

        input_val : 2d array : default None
            the `input_value <Mechanism_Base.input_value>` of the `Mechanism` or `external_input_values
            <Composition.external_input_values>` of the `Composition` for which execution is being reported;
            if it is not specified, it is resolved by calling the node's get_input_values() method.

        params : 'params' or 'parameters' : default None
            specifies whether to report the values of the `Parameters` of the `node <Composition_Nodes>` being executed
            together with its input and output.

        output_val : 2d array : default None
            the `output_values <Mechanism_Base.output_value>` of the `Mechanism` or `external_output_values
            <Composition.external_output_values>` of the `Composition` for which execution is being reported.

        report_output : ReportOutput : default ReportOutput.OFF
            conveys `ReportOutput` option specified in the **report_output** argument of the call to a Composition's
            `execution method <Composition_Execution_Method>` or a Mechanism's `execute <Mechanism_Base.execute>`
            method.

        trial_num : int : default None
            current `TRIAL <TimeScale.TRIAL>` number (None if not known).

        is_controller : bool : default False
            specifies whether or not the node is the `controller <Composition.controller>` of a Composition.

        context : Context : default None
            context of current execution.
        """

        indent = '  '
        if is_controller:
            indent = ''

        depth_indent = 0
        if self._simulating or self._execution_stack_depth:
            depth_indent = self.depth_indent_factor * self._execution_stack_depth

        # Use TERSE format if that has been specified by report_output (i.e., in the arg of an execution method),
        #   or as the reportOutputPref for a node when USE_PREFS is in effect
        node_pref = convert_to_list(node.reportOutputPref)
        # Get reportOutputPref if specified, and remove from node_pref (for param processing below)
        report_output_pref = [node_pref.pop(node_pref.index(pref))
                              for pref in node_pref if isinstance(pref, ReportOutput)]
        node_params_prefs = node_pref
        if (report_output is ReportOutput.TERSE
                or (report_output is not ReportOutput.FULL and ReportOutput.TERSE in report_output_pref)):
            if is_controller:
                execute_str = f'simulation of {node.composition.name} ' \
                              f'[underline]{node.composition.controller_mode}[/] TRIAL {trial_num}'
            else:
                execute_str = 'executed'
            return f'[{node_panel_color}]{depth_indent * " "}{indent}{node.name} {execute_str}'

        # Render input --------------------------------------------------------------------------------------------

        if input_val is None:
            input_val = node.get_input_values(context)
        # FIX: kmantel: previous version would fail on anything but iterables of things that can be cast to floats
        #      if you want more specific output, you can add conditional tests here
        try:
            input_string = [float("{:0.3}".format(float(i))) for i in input_val].__str__().strip("[]")
        except TypeError:
            input_string = input_val

        input_report = f"input: {input_string}"

        # Render output --------------------------------------------------------------------------------------------

        if output_val is None:
            output = node.output_port.parameters.value._get(context)
        # FIX: kmantel: previous version would fail on anything but iterables of things that can be cast to floats
        #   if you want more specific output, you can add conditional tests here
        try:
            # output_string = re.sub(r'[\[,\],\n]', '', str([float("{:0.3}".format(float(i))) for i in output_val]))
            output_string = re.sub(r'[\[,\],\n]', '', str([float("{:0.3}".format(float(i))) for i in output_val]))
        except TypeError:
            output_string = output

        output_report = f"output: {output_string}"

        # Render params if specified -------------------------------------------------------------------------------

        from psyneulink.core.components.shellclasses import Function
        report_params = convert_to_list(report_params)
        params = {p.name: p._get(context) for p in node.parameters}
        try:
            # Check for PARAMS keyword (or 'parameters') and remove from node_prefs if there
            if node_params_prefs and isinstance(node_params_prefs[0],list):
                node_params_prefs = node_params_prefs[0]
            params_keyword = [node_params_prefs.pop(node_params_prefs.index(p))
                              for p in node_params_prefs if re.match('param(eter)?s?', p, flags=re.IGNORECASE)]
            # Get any parameters for the node itself
            node_params = [node_params_prefs.pop(node_params_prefs.index(p))
                              for p in node_params_prefs if p in params]
            # If any are left, assume they are for the node's function
            function_params = node_params_prefs
            # Display parameters if any are specified
            include_params = node_params or function_params or params_keyword or report_params
        except (TypeError, IndexError):
            # include_params = False
            # assert False, f'PROGRAM ERROR: Problem processing reportOutputPref args for {node.name}.'
            raise ReportError(f"Unrecognized specification for reportOutputPref of {node.name}: {node_params_prefs}.")

        params_string = ''
        function_params_string = ''

        if include_params:

            def param_is_specified(name: str, specified_set: list, param_type: str) -> Union[str, bool]:
                """Helper method: check whether param has been specified based on options"""

                from psyneulink.core.components.mechanisms.mechanism import Mechanism
                from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
                from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism \
                    import CompositionInterfaceMechanism
                from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism \
                    import ModulatoryMechanism_Base

                # Helper methods for testing whether param satisfies specifications -----------------------------

                def get_controller(proj) -> Optional[str]:
                    """Helper method: get modulator (controller) of modulated params"""
                    # if not isinstance(proj.sender.owner, CompositionInterfaceMechanism):
                    if isinstance(proj.sender.owner, ModulatoryMechanism_Base):
                        return proj.sender.owner.name
                    #   mediating a projection from a ModulatoryMechanism in a Composition in which this is nested
                    if not isinstance(proj.sender.owner, CompositionInterfaceMechanism):
                        assert False, f'PROGRAM ERROR Projection to ParameterPort for {param_name} of {node.name} ' \
                                      f'from non ModulatoryMechanism'
                    # Recursively call to get ModulatoryMechanism in outer Composition
                    return get_controller(proj.sender.owner.afferents[0])

                def is_modulated() -> Optional[str]:
                    """Helper method: determine whether parameter is being modulated
                       by checking whether ParameterPort receives aControlProjection
                    """
                    try:
                        if isinstance(node, Mechanism):
                            if name in node.parameter_ports.names:
                                param_port = node.parameter_ports[name]
                                if param_port.mod_afferents:
                                    controller_names = [get_controller(c) for c in param_port.mod_afferents]
                                    controllers_str = ' and '.join(controller_names)
                                    return f'modulated by {controllers_str}'
                    except:
                        print(f'Failed to find {name} on {node.name}')

                def get_monitor(proj) -> Union[str,list,None]:
                    """Helper method: get modulator (controller) of modulated params"""
                    # if not isinstance(proj.sender.owner, CompositionInterfaceMechanism):
                    if isinstance(proj.receiver.owner, (ObjectiveMechanism, ModulatoryMechanism_Base)):
                        return proj.receiver.owner.name
                    # Mediating a projection from a monitored Mechanism to a Composition in which it is nested, so
                    # recursively call to get receiver in outer Composition
                    if isinstance(proj.receiver.owner, CompositionInterfaceMechanism) and proj.receiver.owner.efferents:
                        # owners = []
                        # for efferent in proj.receiver.owner.efferents:
                        #     owner = get_monitor(efferent)
                        #     if owner:
                        #         owners.extend(owner)
                        # return(owners)
                        owners = [get_monitor(efferent) for efferent in proj.receiver.owner.efferents]
                        return owners

                def is_monitored() -> Optional[str]:
                    """Helper method: determine whether parameter is being monitored by checking whether OutputPort
                    sends a MappingProjection to an ObjectiveMechanism or  ControlMechanism.
                    """
                    try:
                        # Restrict to looking for VALUE parameter on nodes (i.e., not functions)
                        if name in VALUE and isinstance(node, Mechanism) and param_type is 'node':
                            monitor_names = []
                            for output_port in node.output_ports:
                                monitors = []
                                for proj in output_port.efferents:
                                    monitor = get_monitor(proj)
                                    if isinstance(monitor, list):
                                        monitors.extend(monitor)
                                    else:
                                        monitors.append(monitor)
                                monitor_names.extend([monitor_name for monitor_name in monitors if monitor_name])
                            if monitor_names:
                                monitor_str = ' and '.join(monitor_names)
                                return f'monitored by {monitor_str}'
                    except:
                        print(f'Failed to find {name} on {node.name}')

                def is_logged(node, name):
                    try:
                        if (LogCondition.from_string(node.log.loggable_items[name])
                                & (LogCondition.TRIAL | LogCondition.RUN)):
                            return True
                    except KeyError:
                        pass

                # Evaluate tests: -----------------------------------------------------------------------

                # Get modulated and monitored descriptions if they apply
                mod_str = is_modulated()
                monitor_str = is_monitored()
                if monitor_str and mod_str:
                    control_str = " and ".join([monitor_str, mod_str])
                else:
                    control_str = monitor_str or mod_str
                if control_str:
                    control_str = f' ({control_str})'

                # Include if param is explicitly specified or ReportParams.ALL (or 'params') is specified
                if (name in specified_set
                        # FIX: ADD SUPPORT FOR ReportParams.ALL
                        # PARAMS specified as keyword to display all params
                        or include_params is params_keyword):
                    return control_str or True

                # Include if param is modulated and ReportParams.MODULATED (CONTROLLED) is specified
                if any(k in report_params for k in (ReportParams.MODULATED, ReportParams.CONTROLLED)) and mod_str:
                    return control_str

                # Include if param is monitored and ReportParams.MONITORED is specified
                # FIX: PUT CHECK FOR node EARLIER??
                # if ReportParams.MONITORED in report_params and monitor_str and param_type is 'node':
                if ReportParams.MONITORED in report_params and monitor_str:
                    return control_str

                # Include if param is being logged and ReportParams.LOGGED is specified
                if ReportParams.LOGGED in report_params:
                    # FIX: RESTRICT VALUE AND VARIABLE TO MECHANISM (USE FUNC_VALUE AND FUNC_VARIBALE FOR FUNCTION)
                    return is_logged(node, name)

                return False

            # Test whether param matches specifications: -----------------------------------------

            # Sort for consistency of output
            params_keys_sorted = sorted(params.keys())
            for param_name in params_keys_sorted:

                # Check for function
                param_is_function = False
                # No need to report:
                #    function_params here, as they will be reported for the function itself below;
                #    input_ports or output_ports, as these are inherent in the structure
                if param_name in {FUNCTION_PARAMS, INPUT_PORTS, OUTPUT_PORTS}:
                    continue
                param_value = params[param_name]
                # PsyNeuLink Function
                if isinstance(param_value, Function):
                    param = param_value.name
                    param_is_function = True
                # PsyNeuLink Function class
                elif isinstance(param_value, type(Function)):
                    param = param_value.__name__
                    param_is_function = True
                # Python, Numpy or other type of function
                elif isinstance(param_value, (types.FunctionType, types.MethodType)):
                    param = param_value.__name__

                # Node param(s)
                qualification = param_is_specified(param_name, node_params, param_type='node')
                if qualification:
                    # Put in params_string if param is specified or 'params' is specified
                    param_value = params[param_name]
                    if not params_string:
                        # Add header
                        params_string = (f"params:")
                    param_value_str = str(param_value).__str__().strip('[]')
                    if isinstance(qualification, str):
                        qualification = qualification
                    else:
                        qualification = ''
                    params_string += f"\n\t{param_name}: {param_value_str}{qualification}"
                    if node_params:
                        node_params.pop(node_params.index(param_name))
                    # Don't include functions in params_string yet (to keep at bottom of report)
                    continue

                # Function param(s)
                if param_is_function:
                    # Sort for consistency of output
                    # func_params_keys_sorted = sorted(node.function.parameters.names())
                    func_params_keys_sorted = sorted(getattr(node, param_name).parameters.names())
                    header_printed = False
                    for fct_param_name in func_params_keys_sorted:
                        # Put in function_params_string if function param is specified or 'params' is specified
                        # (appended to params_string below to keep functions at bottom of report)
                        modulated = False
                        qualification = param_is_specified(fct_param_name, function_params, param_type='func')
                        if qualification:
                            if not header_printed:
                                function_params_string += f"\n\t{param_name}: {param_value.name.__str__().strip('[]')}"
                                header_printed = True
                            param_value = getattr(getattr(node,param_name).parameters,fct_param_name)._get(context)
                            param_value = np.squeeze(param_value)
                            param_value_str = str(param_value).__str__().strip('[]')
                            if not params_string:
                                params_string = (f"params:")
                            if isinstance(qualification, str):
                                qualification = qualification
                            else:
                                qualification = ''
                            function_params_string += f"\n\t\t{fct_param_name}: {param_value_str}{qualification}"
                            if function_params:
                                function_params.pop(function_params.index(fct_param_name))

            assert not node_params, f"PROGRAM ERROR in execution of Report.node_execution_report() " \
                                    f"for '{node.name}': {node_params} remaining in node_params."
            if function_params:
                raise ReportError(f"Unrecognized param(s) specified in "
                                  f"reportOutputPref for '{node.name}': '{', '.join(function_params)}'.")

            params_string += function_params_string

        # Generate report -------------------------------------------------------------------------------

        if params_string:
            width = 100
            expand = True
            node_report = RenderGroup(input_report,Panel(params_string), output_report)
        else:
            width = None
            expand = False
            node_report = f'{input_report}\n{output_report}'

        report = Panel(node_report,
                       box=node_panel_box,
                       border_style=node_panel_color,
                       width=width,
                       expand=expand,
                       title=f'[{node_panel_color}]{node.name}',
                       highlight=True
                       )

        # Don't indent for nodes in Panels (Composition.controller is not in a Panel)
        if not is_controller:
            depth_indent = 0

        return Padding.indent(report, depth_indent)

    def _print_and_record_reports(self, report_type:str, context:Context, output_report:OutputReport=None):
        """
        Conveys output reporting to device specified in `_report_to_devices <Report._report_to_devices>`.
        Called by `report_output <Report.report_output>` and `report_progress <Report.report_progress>`

        Arguments
        ---------

        output_report : int
            id of OutputReport for caller[run_mode] in self.output_reports to use for reporting.
        """

        # Print and record output report as they are created (progress reports are printed by _rich_progress.console)
        if report_type in {TRIAL_REPORT, RUN_REPORT}:
            # Print output reports as they are created
            if self._rich_console or self._rich_divert:
                if output_report.trial_report and report_type is TRIAL_REPORT:
                    self._rich_progress.console.print(output_report.trial_report)
                    self._rich_progress.console.print('')
                elif output_report.run_report and report_type is RUN_REPORT:
                    self._rich_progress.console.print(output_report.run_report)
                    self._rich_progress.console.print('')
            # Record output reports as they are created
            if len(self._execution_stack)==1 and self._report_output is not ReportOutput.OFF:
                    if self._rich_divert:
                        self._rich_diverted_reports += (f'\n{self._rich_progress.console.file.getvalue()}')
                    if self._record_reports:
                        with self._recording_console.capture() as capture:
                            if report_type is TRIAL_REPORT:
                                self._recording_console.print(output_report.trial_report)
                            elif report_type is RUN_REPORT:
                                self._recording_console.print(output_report.run_report)
                        self._recorded_reports += capture.get()

        # Record progress after execution of outer-most Composition
        if len(self._execution_stack)==0:
            if report_type is PROGRESS_REPORT:
                # add progress report to any already recorded for output
                progress_reports = '\n'.join([t.description for t in self._rich_progress.tasks])
                if self._rich_divert:
                    self._rich_diverted_reports += progress_reports + '\n'
                if self._record_reports:
                    self._recorded_reports += progress_reports + '\n'
            outer_comp = self._execution_stack[0]
            # store recorded reports on outer-most Composition
            if self._rich_divert:
                outer_comp.rich_diverted_reports = self._rich_diverted_reports
            if self._record_reports:
                outer_comp.recorded_reports = self._recorded_reports

    @property
    def _execution_stack_depth(self):
        return len(self._execution_stack) - 1

    @property
    def _nested(self):
        from psyneulink.core.compositions.composition import Composition
        return any(isinstance(c, Composition) for c in self._execution_stack)

    @property
    def _simulating(self):
        from psyneulink.core.components.mechanisms.mechanism import Mechanism
        return any(isinstance(c, Mechanism) for c in self._execution_stack)
