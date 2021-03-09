import re
import sys
import warnings
from io import StringIO

from rich import print, box
from rich.console import Console, RenderGroup
from rich.panel import Panel
from rich.progress import Progress as RichProgress

from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import CONSOLE, DIVERT, FULL, PNL_VIEW, RECORD, TERSE
from psyneulink.core.globals.utilities import convert_to_list

SIMULATION = 'Simulat'
DEFAULT = 'Execut'
REPORT_REPORT = False # USED FOR DEBUGGING

DEVICE_KEYWORDS = (CONSOLE, DIVERT, RECORD, PNL_VIEW)

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


class ReportError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ProgressReport():
    """
    Object used to package reporting for a call to Composition.run()
    """

    def __init__(self, id, num_trials):
        self.num_trials = num_trials
        self.rich_task_id = id # used for task id in rich
        self.trial_report = []
        self.time_step_report = []


class Report:
    """
    A singleton context object that provides interface to output and progress reporting.
    It returns the currently active progress context instance if one has been instantiated already in another scope.
    It deallocates the progress bar when the outermost context is released.

    Output reports contain information about the input and output to a `Composition` and its `Nodes <Composition_Nodes>`
    during execution; they are constructed as the Components execute.  If **report_output** is specified as True or
    *TERSE*, the information is reported to the the devices specified in `report_to_devices <Report.report_to_devices>`
    as it is generated;  if *FULL* is specified, then the information is reported at the end of each `TRIAL
    <TimeScale.TRIAL>` of execution.

    Progress reports provide information about the status of execution, and are updated at the end of each `TRIAL
    <TimeScale.TRIAL>` of execution.

    Arguments
    ---------

    report_output : bool, *TERSE*, or *FULL* : default False
        specifies whether to show output of the execution on a trial-by-trial as it is generated.  Any one the
        following options can be used:

        * False - no output is generated;
        * True - output is determined by the `reportoutputpref <PreferenceSet_reportOutputPref>` preference of
          individual Components;
        * *TERSE* - a single line is generated reporting the execution of each Component;
        * *FULL* - input and output of all Components being executed is reported.

    report_progress : bool : default False
        specifies whether to report progress of execution in real time.  If the number trials to be executed
        is explicitly specified, the number of trials executed, a progress bar, and time remaining are displayed;
        if the number of trials is not explicitly specified (e.g., if inputs are specified using a generator), then
        a "spinner" is displayed during execution and the the total number of trials executed is displayed once
        complete.  Progress is reported to the devices specified in `report_to_devices <Report.report_to_devices>`.

    report_simulations : bool : default False
        specifies whether to show output and progress for simulations executed by an `OptimizationControlMechanism`.

    report_to_devices : CONSOLE, RECORD, DIVERT, PNL_VIEW or list : default CONSOLE
        specifies where output and progress should be reported;  the following destinations are supported:

        * *CONSOLE* - directs reporting to the Console of the rich Progress object stored in `_instance._rich_progress
          <Report._rich_progress>` (default);
        * *RECORD* - captures reporting in `_recorded_reports <Report._recorded_reports>`; specifying this
          option on its own replaces and suppresses reporting to the console; to continue to generate console
          output, explicitly include *CONSOLE* with *RECORD* in the argument specification.
        * *DIVERT* - captures reporting otherwise directed to the rich Console in a UDF-8 formatted string and
          stores it in `_rich_diverted_reports <Report._rich_diverted_reports>`. This option suppresses
          console output and is cumulative (that is, it records the sequences of updates sent to the console
          after each TRIAL) and is intended primarily for unit testing. The *RECORD* option should be used for
          recording output, as it can be used with console output if desired, and reflects the final state of
          the display after execution is complete.
        * *PNL_VIEW* - directs reporting to the PsyNeuLinkView graphical interface [UNDER DEVELOPMENT].

    Attributes
    ----------

    _instance : Report
        singleton instance of class;  contains attributes for:

        * a rich Progress object (`_rich_progress`)
        * a PsyNeuLinkView interface object contained in `_PNL_View` - TBI.

    _enable_reporting : bool : default False
        determines whether reporting is enabled;  True if either the **_report_output** or **_report_progress**
        progress arguments of the constructor were specified as not False.

    _use_rich : False, *CONSOLE*, *DIVERT* or list: default *CONSOLE*
        identifies whether reporting to rich is enabled (i.e., if *CONSOLE* and/or *DIVERT* were specified in
        **report_to_devices** argument of constructor.

    _rich_console : bool : default True
        determines whether reporting is sent to _rich_progress console;  True if _enable_reporting is True and
        **CONSOLE** was specified in the **report_to_devices** argument of constructor.

    __rich_console_capture : bool : default True
        determines whether reporting is sent to `_rich_console_capture <Report._rich_console_capture>;  True if
        _enable_reporting is True and **DIVERT** was specified in the **report_to_devices** argument of constructor.

    _use_pnl_view : bool : default False
        determines whether reporting is sent to PsyNeuLinkView if _enable_reporting is True - TBI.

    report_to_devices : list
        list of devices currently enabled for reporting.

    _report_output : bool, *TERSE*, or *FULL* : default False
        determines whether and, if so, what form of output is displayed and/or captured.

    _report_progress : bool : default False
        determines whether progress is displayed and/or captured.

    _report_simulations : bool : default False
        determines whether reporting occurs for output and/or progress of simulations carried out by the `controller
        <Composition_Controller>` of a `Composition`.

    _record_reports : bool : default False
        determines whether reporting is recorded in `recorded_reports <Report.recorded_reports>`.

    _progress_reports : dict
        contains entries for each Composition (the key) executed during progress reporting; the value of each
        entry is itself a dict with two entries:
        - one containing ProgressReports for executions in DEFAULT_MODE (key: DEFAULT)
        - one containing ProgressReports for executions in SIMULATION_MODE (key: SIMULATION)

    _recorded_reports : str :  default []
        if _record_reports is True, contains a record of reports generated during execution.

    _rich_console_capture : str :  default []
        if __rich_divert is True, contains output sent to _rich_progress.console.

    _ref_count : int : default 0
        tracks how many times object has been referenced;  counter is incremented on each context __enter__
        and decrements on each __exit__, to ensure stop progress is not called until all references have been released.

    """

    _instance = None


    def __new__(cls,
                report_progress:bool=False,
                report_output:bool=False,
                report_simulations:bool=False,
                report_to_devices:(*DEVICE_KEYWORDS, list)=CONSOLE
                ) -> 'Report':
        if cls._instance is None:
            cls._instance = super(Report, cls).__new__(cls)

            # cls._enable_progress = report_progress
            cls._report_progress = report_progress
            cls._report_output = report_output
            cls._enable_reporting = report_output or report_progress

            cls._report_to_devices = convert_to_list(report_to_devices or CONSOLE)
            if not any(a in [CONSOLE, DIVERT, RECORD, PNL_VIEW] for a in cls._report_to_devices):
                raise ReportError(f"Unrecognized keyword in argument for 'report_to_devices'; "
                                  f"must be one of: {DEVICE_KEYWORDS}")
            cls._rich_console = CONSOLE in cls._report_to_devices
            cls._rich_divert = DIVERT in cls._report_to_devices
            cls._record_reports = RECORD in cls._report_to_devices
            # Enable rich if reporting output or progress and using console or recording
            cls._use_rich = ((report_output or report_progress)
                             and (cls._rich_console or cls._rich_divert or cls._record_reports))
            cls._use_pnl_view = PNL_VIEW in cls._report_to_devices

            # Show simulations if specified
            cls._report_simulations = report_simulations
            cls._prev_simulation = False

            # Instantiate rich progress context object
            # - it is not started until the self.start_progress_report() method is called
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

            cls._progress_reports = {}
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

    def start_progress_report(self, comp, num_trials, context) -> int:

        # Generate space before beginning of output
        if self._use_rich and not self._progress_reports:
            print()

        if comp not in self._progress_reports:
            self._progress_reports.update({comp:{DEFAULT:[], SIMULATION:[]}})

        # Used for accessing progress report and reporting results
        if context.runmode & ContextFlags.SIMULATION_MODE:
            run_mode = SIMULATION
        else:
            run_mode = DEFAULT

        if run_mode is SIMULATION and not self._report_simulations:
            return

        # Don't create a new report for simulations in a set
        if run_mode is SIMULATION and self._prev_simulation:
            return len(self._progress_reports[comp][run_mode]) - 1

        if self._use_rich:

            # visible = self._report_progress and (run_mode is not SIMULATION or self._report_simulations)
            visible = (self._rich_console
                       and self._report_progress
                       and (run_mode is not SIMULATION or self._report_simulations)
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

            id = self._rich_progress.add_task(f"[red]{run_mode}ing {comp.name}...",
                                         total=num_trials,
                                         start=start,
                                         visible=visible
                                         )

            self._progress_reports[comp][run_mode].append(ProgressReport(id, num_trials))
            report_num = len(self._progress_reports[comp][run_mode]) - 1

            self._prev_simulation = run_mode is SIMULATION

            return report_num

    def report_progress(self, caller, report_num, context):

        if not self._report_progress:
            return

        simulation_mode = context.runmode & ContextFlags.SIMULATION_MODE
        if simulation_mode:
            run_mode = SIMULATION
        else:
            run_mode = DEFAULT

        # Return if (nested within) a simulation and not reporting simulations
        if run_mode is SIMULATION and not self._report_simulations:
            return

        progress_report = self._progress_reports[caller][run_mode][report_num]
        trial_num = self._rich_progress.tasks[progress_report.rich_task_id].completed

        # Useful for debugging:
        if caller.verbosePref or REPORT_REPORT:
            from pprint import pprint
            pprint(f'{caller.name} {str(context.runmode)} REPORT')

        # Update progress report
        if self._use_rich:
            if progress_report.num_trials:
                if simulation_mode:
                    num_trials_str = ''
                else:
                    num_trials_str = f' of {progress_report.num_trials}'
            else:
                num_trials_str = ''

            update = f'{caller.name}: {run_mode}ed {trial_num+1}{num_trials_str} trials'
            self._rich_progress.update(progress_report.rich_task_id,
                                  description=update,
                                  advance=1,
                                  refresh=True)

        # track number of outer (non-simulation) trials
        if (not simulation_mode
                and progress_report.num_trials
                and (trial_num == progress_report.num_trials)):
            self._progress_reports[caller][run_mode].pop()

    def report_output(self, caller,
                      report_num,
                      scheduler,
                      report_output,
                      content,
                      context,
                      nodes_to_report=False,
                      node=None):

        if report_num is None or (not report_output and not self._report_simulations):
            return
        # if report_output is None, defer to Composition's reportOutputPref
        if report_output is None:  # if it is False, leave as is to suppress output
            report_output = caller.reportOutputPref

        if report_output is FULL:   # give precedence to argument in call to execute
            report_type = FULL
        elif report_output is TERSE:
            report_type = TERSE
        else:
            report_type = None

        simulation_mode = context.runmode & ContextFlags.SIMULATION_MODE
        if simulation_mode:
            run_mode = SIMULATION
            sim_str = ' SIMULATION'
        else:
            run_mode = DEFAULT
            sim_str = ''

        progress_report = self._progress_reports[caller][run_mode][report_num]

        trial_num = scheduler.clock.time.trial

        if content is 'trial_init':

            progress_report.trial_report = []

            if report_output is not False or self._report_simulations is not False:  # if it is False, suppress output
                report_output = report_output or caller.reportOutputPref # if it is None, defer to Composition's
                # reportOutputPref

                #  if FULL output, report trial number and Composition's input
                #  note:  header for Trial Panel is constructed under 'content is Trial' case below
                if report_type is FULL:
                    progress_report.trial_report = [f"\n[bold {trial_panel_color}]input:[/]"
                                                     f" {[i.tolist() for i in caller.get_input_values(context)]}"]
                else: # TERSE output
                    # print trial title and separator + input array to Composition
                    trial_header = f"[bold {trial_panel_color}]{caller.name}{sim_str} TRIAL {trial_num} " \
                                   f"===================="
                    self._rich_progress.console.print(trial_header)
                    if self._record_reports:
                        self._recorded_reports += trial_header

        elif content is 'time_step_init':
            if report_output or self._report_simulations:
                if report_type is FULL:
                    progress_report.time_step_report = [] # Contains rich.Panel for each node executed in time_step
                elif nodes_to_report:
                    time_step_header = f'[{time_step_panel_color}] Time Step {scheduler.clock.time.time_step} ---------'
                    self._rich_progress.console.print(time_step_header)
                    if self._record_reports:
                        self._recorded_reports += time_step_header

        elif content is 'node':
            if not node:
                assert False, 'Node not specified in call to Report report_output'
            if ((report_output is False and self._report_simulations is False)
                    or report_output is True and node.reportOutputPref is False):
                return
            # Use FULL node report for Node:
            if report_type is FULL or node.reportOutputPref in [True, FULL]:
                node_report = self.node_execution_report(node,
                                                         input_val=node.get_input_values(context),
                                                         output_val=node.output_port.parameters.value._get(context),
                                                         report_output=report_output,
                                                         context=context
                                                         )
                # If trial is using FULL report, save Node's to progress_report
                if report_type is FULL:
                    progress_report.time_step_report.append(node_report)
                # Otherwise, just print it to the console (as part of otherwise TERSE report)
                else:
                    self._rich_progress.console.print(node_report)
                    if self._record_reports:
                        with self._recording_console.capture() as capture:
                            self._recording_console.print(node_report)
                        self._recorded_reports += capture.get()
            # Use TERSE report for Node
            else:
                self._rich_progress.console.print(f'[{node_panel_color}]  {node.name} executed')

        elif content is 'time_step':
            if (report_output and (nodes_to_report or report_output is FULL) and report_type is FULL):
                progress_report.trial_report.append('')
                progress_report.trial_report.append(Panel(RenderGroup(*progress_report.time_step_report),
                                                           # box=box.HEAVY,
                                                           border_style=time_step_panel_color,
                                                           box=time_step_panel_box,
                                                           title=f'[bold {time_step_panel_color}]\nTime Step '
                                                                 f'{scheduler.clock.time.time_step}[/]',
                                                           expand=False))

        elif content is 'trial':
            if report_type is FULL:
                output_values = []
                for port in caller.output_CIM.output_ports:
                    output_values.append(port.parameters.value._get(context))
                progress_report.trial_report.append(f"\n[bold {trial_output_color}]result:[/]"
                                          f" {[r.tolist() for r in output_values]}\n")
                progress_report.trial_report = Panel(RenderGroup(*progress_report.trial_report),
                                                     box=trial_panel_box,
                                                     border_style=trial_panel_color,
                                                     title=f'[bold{trial_panel_color}] {caller.name}{sim_str}: '
                                                           f'Trial {trial_num} [/]',
                                                     expand=False)
            # FIX: THIS GENERATES A CUMULATIVE REPORT, BUT COMMENTING IT OUT ELIMINATES THE OUTPUT REPORT
            # elif self._rich_divert:
            #     self._rich_diverted_reports += f'\n{self._rich_progress.console.file.getvalue()}'
            # elif report_type is TERSE:
            #     progress_report.trial_report.append(f'\n{self._rich_progress.console.file.getvalue()}')

            # If execute() was called from COMMAND_LINE (rather than via run()), report progress
            if context.source & ContextFlags.COMMAND_LINE and (report_output or self._report_simulations):
                self._print_reports(progress_report)

        elif content is 'run':
            if report_output or self._report_simulations:
                self._print_reports(progress_report)

        return

    def _print_reports(self, progress_report):
        # if self._rich_console and progress_report.trial_report:
        if (self._rich_console or self._rich_divert) and progress_report.trial_report:
            self._rich_progress.console.print(progress_report.trial_report)
            self._rich_progress.console.print('')
        update = '\n'.join([t.description for t in self._rich_progress.tasks])
        if self._report_output:
            if self._rich_divert:
                self._rich_diverted_reports += (f'\n{self._rich_progress.console.file.getvalue()}')
            if self._record_reports:
                with self._recording_console.capture() as capture:
                    self._recording_console.print(progress_report.trial_report)
                self._recorded_reports += capture.get()
        if self._report_progress:
            if self._rich_divert:
                self._rich_diverted_reports += update + '\n'
            if self._record_reports:
                self._recorded_reports += update + '\n'

        assert True

    @staticmethod
    def node_execution_report(node,
                              input_val=None,
                              params=None,
                              output_val=None,
                              report_output=True,
                              context=None):
        from psyneulink.core.components.shellclasses import Function
        from psyneulink.core.globals.keywords import FUNCTION_PARAMS

        node_report = ''

        if report_output is TERSE or node.reportOutputPref is TERSE and report_output is not FULL:
            return f'[{node_panel_color}]{node.name} executed'

        if input_val is None:
            input_val = node.get_input_values(context)
        if output_val is None:
            output = node.output_port.parameters.value._get(context)
        params = params or {p.name: p._get(context) for p in node.parameters}

        # print input
        # FIX: kmantel: previous version would fail on anything but iterables of things that can be cast to floats
        #      if you want more specific output, you can add conditional tests here
        try:
            input_string = [float("{:0.3}".format(float(i))) for i in input_val].__str__().strip("[]")
        except TypeError:
            input_string = input_val

        node_report += f"input: {input_string}"

        # print output
        # FIX: kmantel: previous version would fail on anything but iterables of things that can be cast to floats
        #   if you want more specific output, you can add conditional tests here
        try:
            output_string = re.sub(r'[\[,\],\n]', '', str([float("{:0.3}".format(float(i))) for i in output_val]))
        except TypeError:
            output_string = output

        node_report += f"\noutput: {output_string}"

        # print params
        try:
            include_params = re.match('param(eter)?s?', node.reportOutputPref, flags=re.IGNORECASE)
        except TypeError:
            include_params = False

        if include_params:
            # print("- params:")
            params_string = (f"\n- params:")
            # Sort for consistency of output
            params_keys_sorted = sorted(params.keys())
            for param_name in params_keys_sorted:
                # No need to report:
                #    function_params here, as they will be reported for the function itself below;
                #    input_ports or output_ports, as these are inherent in the structure
                if param_name in {FUNCTION_PARAMS, INPUT_PORTS, OUTPUT_PORTS}:
                    continue
                param_is_function = False
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
                else:
                    param = param_value
                params_string += f"\n\t{param_name}: {str(param).__str__().strip('[]')}"
                if param_is_function:
                    # Sort for consistency of output
                    func_params_keys_sorted = sorted(node.function.parameters.names())
                    for fct_param_name in func_params_keys_sorted:
                        params_string += ("\n\t\t{}: {}".
                                          format(fct_param_name,
                                                 str(getattr(node.function.parameters, fct_param_name)._get(context)).__str__().strip("[]")))

        if include_params:
            width = 100
            expand = True
            node_report = RenderGroup(node_report,Panel(params_string))
            params_string
        else:
            width = None
            expand = False
        return Panel(node_report,
                     box=node_panel_box,
                     border_style=node_panel_color,
                     width=width,
                     expand=expand,
                     title=f'[{node_panel_color}]{node.name}',
                     highlight=True)
