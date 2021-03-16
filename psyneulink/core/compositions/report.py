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
and/or the `reportOutputPref <PreferenceSet_reportOutputPref>` settings of individual Mechanisms. Whether `simulations
<OptimizationControlMechanism_Execution>` executed by a Composition's `controller <Composition_Controller>` are
included is determined by the **report_simulations** argument using a `ReportSimulations` option. Output is reported
to the devices specified in the **report_to_devices** argument using the `ReportDevices` options.


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

import numpy as np
from rich import print, box
from rich.console import Console, RenderGroup
from rich.panel import Panel
from rich.progress import Progress as RichProgress

from psyneulink.core.globals.context import Context
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import FUNCTION_PARAMS, INPUT_PORTS, OUTPUT_PORTS, VALUE
from psyneulink.core.globals.utilities import convert_to_list

__all__ = ['Report', 'ReportOutput', 'ReportParams', 'ReportProgress', 'ReportDevices', 'ReportSimulations',
           'CONSOLE', 'CONTROLLED', 'LOGGED', 'MODULATED', 'RECORD', 'DIVERT', 'PNL_VIEW', ]

SIMULATION = 'Simulat'
DEFAULT = 'Execut'
SIMULATIONS = 'simulations'
SIMULATING = 'simulating'
REPORT_REPORT = False # USED FOR DEBUGGING
OUTPUT_REPORT = 'output_report'
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


class ReportOutput(Enum):
    """
    Options used in the **report_output** argument of a `Composition`\'s `execution methods
    <Composition_Execution_Methods>` or the `execute <Mechanism_Base.execute>` method of a `Mechanism`, to enable and
    determine the type of output generated by reporting; see `Report_Output` for additional details.

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
    <Composition_Execution_Methods>`, to specify the scope of reporting for its `Parameters` and those of its
    `Nodes <Composition_Nodes>`; see `Reporting Parameter values <Report_Params>` under `Report_Output` for
    additional details.

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
        receives a `ControlProjection` from a `ControlSignal`.

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
LOGGED = ReportParams.LOGGED
ALL = ReportParams.ALL


class ReportProgress(Enum):
    """
    Options used in the **report_progress** argument of a `Composition`\'s `run <Composition.run>` and `learn
    <Composition.learn>` methods, to enable/disable progress reporting during execution of a Composition; see
    `Report_Progress` for additional details.

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
    executed by an a Composition's `controller <Composition_Controller>` are included in output and progress reporting.

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
    <Composition_Execution_Methods>` or the `execute <Mechanism_Base.execute>` method of a `Mechanism`,
    to determine the devices to which reporting is directed.

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


class RunReport():
    """
    Object used to package Progress reporting for a call to the `run <Composition.run>` or `learn
    <Composition.learn>` methods of a `Composition`.
    """

    def __init__(self, id, num_trials):
        self.num_trials = num_trials
        self.rich_task_id = id # used for task id in rich
        self.trial_report = []
        self.time_step_report = []


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

    _report_params : list[ReportParams] : default [ReportParams.USE_PREFS]
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

    _run_reports : dict
        contains entries for each Composition (the key) executed during progress reporting; the value of each
        entry is itself a dict with two entries:
        - one containing RunReports for executions in DEFAULT_MODE (key: DEFAULT)
        - one containing RunReports for executions in SIMULATION_MODE (key: SIMULATION)

    _execution_stack : list : default []
        tracks `nested compositions <Composition_Nested>` and `controllers <Composition_Controller>`
        (i.e., being used to `simulate <OptimizationControlMechanism_Execution>` a `Composition`).

    _execution_stack_depth : int : default 0
        depth of nesting of executions, including `nested compositions <Composition_Nested>` and any `controllers
        <Composition_Controller>` currently executing `simulations <OptimizationControlMechanism_Execution>`.

    _nested_comps : bool : default False
        True if there are any `nested compositions <Composition_Nested>`
        in `_execution_stack <Report._execution_stack>`.

    _simulating : bool : default False
        True if there are any `controllers <Composition_Controller>`
        in `_execution_stack <Report._execution_stack>`.

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
                context:Context=None
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

            cls._execution_stack = [caller]

            # Instantiate rich progress context object
            # - it is not started until the self.start_run_report() method is called
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

            cls._run_reports = {}
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

    def start_run_report(self, comp, num_trials, context) -> int:
        """
        Initialize a RunReport for Composition

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

        RunReport id : int
            id is stored in `_run_reports <Report._run_reports>`.

        """

        if not comp:
            assert False, "Report.start_progress() called without a Composition specified in 'comp'."
        if num_trials is None:
            assert False, "Report.start_progress() called with num_trials unspecified."


        # Generate space before beginning of output
        if self._use_rich and not self._run_reports:
            print()

        if comp not in self._run_reports:
            self._run_reports.update({comp:{DEFAULT:[], SIMULATION:[], SIMULATING:False}})

        # Used for accessing progress report and reporting results
        if context.runmode & ContextFlags.SIMULATION_MODE:
            run_mode = SIMULATION
        else:
            run_mode = DEFAULT

        if run_mode is SIMULATION and self._report_simulations is not ReportSimulations.ON:
            return

        if run_mode is SIMULATION:
            # If already simulating, return existing report for those simulations
            if self._run_reports[comp][SIMULATING]:
                return len(self._run_reports[comp][run_mode]) - 1

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

            indent_factor = 2
            depth_indent = depth_str = ''
            if run_mode is SIMULATION or self._execution_stack_depth:
                depth_indent = indent_factor * self._execution_stack_depth * ' '
                depth_str = f' (depth: {self._execution_stack_depth})'

            id = self._rich_progress.add_task(f"[red]{depth_indent}{comp.name}: {run_mode}ing {depth_str}...",
                                         total=num_trials,
                                         start=start,
                                         visible=visible
                                         )

            self._run_reports[comp][run_mode].append(RunReport(id, num_trials))
            report_num = len(self._run_reports[comp][run_mode]) - 1

            self._run_reports[comp][SIMULATING] = run_mode is SIMULATION

            return report_num

    def report_progress(self, caller, report_num:int, context:Context):
        """
        Report progress of executions in call to `run <Composition.run>` or `learn <Composition.learn>` method of
        a `Composition`, and record reports if specified.

        Arguments
        ---------

        caller : Composition or Mechanism

        report_num : int
            id of RunReport for caller[run_mode] in self._run_reports to use for reporting.

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

        run_report = self._run_reports[caller][run_mode][report_num]
        trial_num = self._rich_progress.tasks[run_report.rich_task_id].completed

        # Useful for debugging:
        if caller.verbosePref or REPORT_REPORT:
            from pprint import pprint
            pprint(f'{caller.name} {str(context.runmode)} REPORT')

        # Not in simulation and have reached num_trials (if specified),
        if self._run_reports[caller][SIMULATING] and not simulation_mode:
            # If was simulating previously, then have just exited, so:
            #   (note: need to use transition and not explicit count of simulations,
            #    since number of simulation trials being run is generally not known)
                # - turn it off
            self._run_reports[caller][SIMULATING] = False

        # Update progress report
        if self._use_rich:
            if run_report.num_trials:
                if simulation_mode:
                    num_trials_str = ''
                else:
                    num_trials_str = f' of {run_report.num_trials}'
            else:
                num_trials_str = ''

            # Construct update text
            indent_factor = 2
            depth_indent = depth_str = ''
            if simulation_mode or self._execution_stack_depth:
                depth_indent = indent_factor * self._execution_stack_depth * ' '
                depth_str = f' (depth: {self._execution_stack_depth})'
            update = f'{depth_indent}{caller.name}: {run_mode}ed {trial_num+1}{num_trials_str} trials{depth_str}'

            # Do update
            self._rich_progress.update(run_report.rich_task_id,
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
        Mechanism <Mechanism_Base.execute>`.

        Arguments
        ---------

        caller : Composition or Mechanism
            Component requesting report;  used to identify relevant run_report.

        report_num : int
            specifies id of `RunReport`, stored in `_run_reports <Report._run_reports>` for each
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
            'time_step', 'trial' or 'run'.

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

        # Determine report type and relevant parameters ----------------------------------------------------------------

        # Get ReportOutputPref for node
        if node:
            node_pref = next((pref for pref in convert_to_list(node.reportOutputPref)
                                         if isinstance(pref, ReportOutput)), None)

        # Assign report_output as default for trial_report_type and node_report_type...
        trial_report_type = node_report_type = report_output

        # then try to get them from caller, based on whether it is a Mechanism or Composition
        from psyneulink.core.compositions.composition import Composition
        from psyneulink.core.components.mechanisms.mechanism import Mechanism
        # Report is called for by a Mechanism
        if isinstance(caller, Mechanism):
            if context.source & ContextFlags.COMPOSITION:
                run_report_owner = context.composition
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
            trial_num = scheduler.clock.time.trial
            run_report_owner = caller

        # Determine run_mode and get run_report if call is from a Composition or a Mechanism being executed by one
        if isinstance(caller, Composition) or context.source == ContextFlags.COMPOSITION:
            simulation_mode = context.runmode & ContextFlags.SIMULATION_MODE
            if simulation_mode and self._report_simulations is ReportSimulations.OFF:
                return
            if simulation_mode:
                run_mode = SIMULATION
                sim_str = ' SIMULATION'
            else:
                run_mode = DEFAULT
                sim_str = ''
            run_report = self._run_reports[run_report_owner][run_mode][report_num]

        # Construct output report -----------------------------------------------------------------------------

        if content is 'trial_init':
            run_report.trial_report = []
            #  if FULL output, report trial number and Composition's input
            #  note:  header for Trial Panel is constructed under 'content is Trial' case below
            if trial_report_type is ReportOutput.FULL:
                run_report.trial_report = [f"\n[bold {trial_panel_color}]input:[/]"
                                                 f" {[i.tolist() for i in caller.get_input_values(context)]}"]
            else: # TERSE output
                # print trial title and separator + input array to Composition
                trial_header = f"[bold {trial_panel_color}]{caller.name}{sim_str} TRIAL {trial_num} " \
                               f"===================="
                self._rich_progress.console.print(trial_header)
                if self._record_reports:
                    self._recorded_reports += trial_header

        elif content is 'time_step_init':
            if trial_report_type is ReportOutput.FULL:
                run_report.time_step_report = [] # Contains rich.Panel for each node executed in time_step
            elif nodes_to_report: # TERSE output
                time_step_header = f'[{time_step_panel_color}] Time Step {scheduler.clock.time.time_step} ---------'
                self._rich_progress.console.print(time_step_header)
                if self._record_reports:
                    self._recorded_reports += time_step_header

        elif content is 'node':
            if not node:
                assert False, 'Node not specified in call to Report report_output'
            node_report = self.node_execution_report(node,
                                                     input_val=node.get_input_values(context),
                                                     output_val=node.output_port.parameters.value._get(context),
                                                     report_output=node_report_type,
                                                     report_params=report_params,
                                                     context=context
                                                     )
            # If trial is using FULL report, save Node's to run_report
            if trial_report_type is ReportOutput.FULL:
                run_report.time_step_report.append(node_report)
            # Otherwise, just print it to the console (as part of otherwise TERSE report)
            else: # TERSE output
                self._rich_progress.console.print(node_report)
                if self._record_reports:
                    with self._recording_console.capture() as capture:
                        self._recording_console.print(node_report)
                    self._recorded_reports += capture.get()

        elif content is 'time_step':
            if nodes_to_report and trial_report_type is ReportOutput.FULL:
                run_report.trial_report.append('')
                run_report.trial_report.append(Panel(RenderGroup(*run_report.time_step_report),
                                                           # box=box.HEAVY,
                                                           border_style=time_step_panel_color,
                                                           box=time_step_panel_box,
                                                           title=f'[bold {time_step_panel_color}]\nTime Step '
                                                                 f'{scheduler.clock.time.time_step}[/]',
                                                           expand=False))

        elif content is 'trial':
            if trial_report_type is ReportOutput.FULL:
                output_values = []
                for port in caller.output_CIM.output_ports:
                    output_values.append(port.parameters.value._get(context))
                run_report.trial_report.append(f"\n[bold {trial_output_color}]result:[/]"
                                          f" {[r.tolist() for r in output_values]}\n")
                run_report.trial_report = Panel(RenderGroup(*run_report.trial_report),
                                                     box=trial_panel_box,
                                                     border_style=trial_panel_color,
                                                     title=f'[bold{trial_panel_color}] {caller.name}{sim_str}: '
                                                           f'Trial {trial_num} [/]',
                                                     expand=False)

            if trial_report_type is not ReportOutput.OFF:
                self._print_and_record_reports(OUTPUT_REPORT, context, run_report)

        else:
            assert False, f"Bad 'content' argument in call to Report.report_output() for {caller.name}: {content}."

        return

    def _print_and_record_reports(self, report_type:str, context:Context, run_report:RunReport=None):
        """
        Conveys output reporting to device specified in `_report_to_devices <Report._report_to_devices>`.
        Called by `report_output <Report.report_output>` and `report_progress <Report.report_progress>`

        Arguments
        ---------

        run_report : int
            id of RunReport for caller[run_mode] in self._run_reports to use for reporting.
        """

        # Print and record output report as they are created (progress reports are printed by _rich_progress.console)
        if report_type is OUTPUT_REPORT:
            # Print output reports as they are created
            if (self._rich_console or self._rich_divert) and run_report.trial_report:
                self._rich_progress.console.print(run_report.trial_report)
                self._rich_progress.console.print('')
            # Record output reports as they are created
            if len(self._execution_stack)==1 and self._report_output is not ReportOutput.OFF:
                    if self._rich_divert:
                        self._rich_diverted_reports += (f'\n{self._rich_progress.console.file.getvalue()}')
                    if self._record_reports:
                        with self._recording_console.capture() as capture:
                            self._recording_console.print(run_report.trial_report)
                        self._recorded_reports += capture.get()

        # Record progress after execution of outer-most Composition
        if len(self._execution_stack)==1:
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

    def node_execution_report(self,
                              node,
                              input_val=None,
                              output_val=None,
                              report_output=ReportOutput.USE_PREFS,
                              report_params=ReportParams.OFF,
                              context=None):
        """
        Generates formatted output report for the `node <Composition_Nodes>` of a `Composition` or a `Mechanism`.
        Called by `report_output <Report.report_output>` for execution of a Composition, and directly by the `execute
        <Mechanism_Base>` method of a `Mechanism` when executed on its own.

        Allows user to specify *PARAMS* or 'parameters' to induce reporting of all parameters, and a listing of
        individual parameters to list just those.

        Arguments
        ---------

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

        report_output : ReportOutput
            conveys `ReportOutput` option specified in the **report_output** argument of the call to a Composition's
            `execution method <Composition_Execution_Method>` or a Mechanism's `execute <Mechanism_Base.execute>`
            method.

        context : Context
            context of current execution.
        """

        # Use TERSE format if that has been specified by report_output (i.e., in the arg of an execution method),
        #   or as the reportOutputPref for a node when USE_PREFS is in effect
        node_pref = convert_to_list(node.reportOutputPref)
        # Get reportOutputPref if specified, and remove from node_pref (for param processing below)
        report_output_pref = [node_pref.pop(node_pref.index(pref))
                              for pref in node_pref if isinstance(pref, ReportOutput)]
        node_params_prefs = node_pref
        if (report_output is ReportOutput.TERSE
                or (report_output is not ReportOutput.FULL and ReportOutput.TERSE in report_output_pref)):
            return f'[{node_panel_color}]  {node.name} executed'

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
            assert False, f'PROGRAM ERROR: Problem processing reportOutputPref args for {node.name}.'
            # include_params = False

        params_string = ''
        function_params_string = ''

        if include_params:

            def param_is_specified(name, specified_set):
                """Helper method: check whether param has been specified based on options"""

                from psyneulink.core.components.mechanisms.mechanism import Mechanism
                from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
                from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism \
                    import CompositionInterfaceMechanism
                from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism \
                    import ModulatoryMechanism_Base

                # Helper methods for testing whether param satisfies specifications -----------------------------

                def get_controller(proj):
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

                def is_modulated():
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
                    # return ''


                def get_monitor(proj):
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


                def is_monitored():
                    """Helper method: determine whether parameter is being monitored by checking whether OutputPort
                    sends a MappingProjection to an ObjectiveMechanism or  ControlMechanism.
                    """
                    try:
                        if name is VALUE and isinstance(node, Mechanism):
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
                    # return ''

                def is_logged():
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

                # FIX: NEED TO FILTER OUT RESPONSES TO FUNCTION VALUE AND TO OBJECTIVE MECHANISM ISELF
                # # Include if param is monitored and ReportParams.MONITORED is specified
                # if ReportParams.MONITORED in report_params and monitor_str:
                #     return control_str

                # Include if param is being logged and ReportParams.LOGGED is specified
                if report_params is ReportParams.LOGGED:
                    pass

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
                if isinstance(param_value, Function):
                    param = param_value.name
                    param_is_function = True
                elif isinstance(param_value, type(Function)):
                    param = param_value.__name__
                    param_is_function = True
                elif isinstance(param_value, (types.FunctionType, types.MethodType)):
                    param = param_value.__node__.__class__.__name__
                    param_is_function = True

                # Node param(s)
                elif param_is_specified(param_name, node_params):
                    # Put in params_string if param is specified or 'params' is specified
                    param_value = params[param_name]
                    if not params_string:
                        # Add header
                        params_string = (f"params:")
                    params_string += f"\n\t{param_name}: {str(param_value).__str__().strip('[]')}"
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
                        qualification = param_is_specified(fct_param_name, function_params)
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

        return Panel(node_report,
                     box=node_panel_box,
                     border_style=node_panel_color,
                     width=width,
                     expand=expand,
                     title=f'[{node_panel_color}]{node.name}',
                     highlight=True)

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
