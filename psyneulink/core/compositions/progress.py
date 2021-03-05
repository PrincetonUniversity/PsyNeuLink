import re
import sys
import warnings
from io import StringIO

from rich import print, box
from rich.console import Console, RenderGroup
from rich.panel import Panel
from rich.progress import Progress as RichProgress

from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import CONSOLE, FILE, FULL, PNL_VIEW, SIMULATIONS, TERSE
from psyneulink.core.globals.utilities import convert_to_list

SIMULATION = 'Simulat'
DEFAULT = 'Execut'
REPORT_REPORT = False # USED FOR DEBUGGING

# rich console report styles
# node
node_panel_color = 'orange1'
# node_panel_box = box.SIMPLE
node_panel_box = box.ROUNDED
# time_step
time_step_panel_color = 'dodger_blue2'
time_step_panel_box = box.SQUARE
# trial
trial_panel_color = 'dodger_blue3'
trial_input_color = 'green'
trial_output_color = 'red'
trial_panel_box = box.HEAVY


class PNLProgressError(Exception):

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


class PNLProgress:
    """
    A singleton context object that provides interface to output and progress reporting (e.g., rich and pnl_view)
    It returns the currently active progress context instance if one has been instantiated already in another scope.
    It deallocates the progress bar when the outermost context is released.

    Arguments
    ---------

    show_progress : bool, CONSOLE, PNL_VIEW, SIMULATIONS, or list : default False
        specifies whether to show progress of execution in real time.  If the number trials to be
        executed is explicitly specified, the number of trials executed, a progress bar, and time remaining are
        displayed; if the number of trials is not explicitly specified (e.g., if inputs are specified using a
        generator), then a "spinner" is displayed during execution and the the total number of trials executed is
        displayed once complete.  The following options can be used to specify what and where the information is
        displayed, either individually or in a list:

        * *SIMULATIONS* - reports simulations executed by an `OptimizationControlMechanism`.

        * *CONSOLE* - directs output to the console (default)

        * *PNL_VIEW* - directs output to the PsyNeuLinkView graphical interface [UNDER DEVELOPMENT]

    Attributes
    ----------

    _instance : PNLProgress
        singleton instance of class.

    _show_progress : bool : default False
        determines whether progress reporting is enabled.

    _use_rich : bool : default True
        determines whether reporting is sent to rich console.

    _use_pnl_view : bool : default False
        determines whether reporting is sent to PsyNeuLinkView - TBI.

    _show_simulations : bool : default False
        determines whether reporting generated for simulations.

    _progress_reports : dict
        contains entries for each Composition (the key) executed during progress reporting; the value of each
        entry is itself a dict with two entries:
        - one containing ProgressReports for executions in DEFAULT_MODE (key: DEFAULT)
        - one containing ProgressReports for executions in SIMULATION_MODE (key: SIMULATION)

    _ref_count : int : default 0
        tracks how many times object has been referenced;  counter is incremented on each context __enter__
        and decrements on each __exit__, to ensure stop progress is not called until all references have been released.

    """

    _instance = None

    def __new__(cls, show_progress=False, show_output=False) -> 'PNLProgress':
        if cls._instance is None:
            cls._instance = super(PNLProgress, cls).__new__(cls)

            cls._show_progress = bool(show_progress)

            show_progress = convert_to_list(show_progress)
            # Use rich console output by default
            cls._use_rich = (False not in show_progress and [k in show_progress for k in {True, CONSOLE, FILE}]
                             or show_output)
            # TBI: send output to PsyNeuLinkView
            cls._use_pnl_view = False not in show_progress and PNL_VIEW in show_progress
            # Show simulations if specified
            cls._show_simulations = False not in show_progress and SIMULATIONS in show_progress

            cls._prev_simulation = False

            # Instantiate rich progress context object
            # - it is not started until the self.start_progress_report() method is called
            # - auto_refresh is disabled to accommodate IDEs (such as PyCharm and Jupyter Notebooks)
            if cls._use_rich:
                file = False
                if FILE in show_progress:
                    file = StringIO()
                # cls._instance._rich_progress = RichProgress(auto_refresh=False)
                cls._instance._rich_progress = RichProgress(auto_refresh=False, console=Console(file=file))

            # Instantiate interface to PsyNeuLinkView
            if cls._use_pnl_view:
                warnings.warn("'pnl_view' not yet supported as an option for show_progress of Composition.run()")

            cls._progress_reports = {}

            cls._ref_count = 0

        return cls._instance

    @classmethod
    def _destroy(cls) -> None:
        """
        A simple helper method that deallocates the singleton instance. This is called when we want to fully destroy
        the singleton instance and its member progress counters. This will cause the next call to PNLProgress() to
        create a completely new singleton instance.
        """
        cls._instance = None

    def __enter__(self):
        """
        This  returns a singleton of the PNLProgress class.
        Returns:
            A new singleton PNL progress context if none is currently active, otherwise, it returns the currently
            active context.
        """

        # If this is the top level call to with PNLProgress(), start progress reporting
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
            PNLProgress._destroy()

    def start_progress_report(self, comp, num_trials, context):

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

        if run_mode is SIMULATION and not self._show_simulations:
            return

        # Don't create a new report for simulations in a set
        if run_mode is SIMULATION and self._prev_simulation:
            return len(self._progress_reports[comp][run_mode]) - 1

        if self._use_rich:

            visible = self._show_progress and (run_mode is not SIMULATION or self._show_simulations)

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

        if not self._show_progress:
            return

        simulation_mode = context.runmode & ContextFlags.SIMULATION_MODE
        if simulation_mode:
            run_mode = SIMULATION
        else:
            run_mode = DEFAULT

        # Return if (nested within) a simulation and not reporting simulations
        if run_mode is SIMULATION and not self._show_simulations:
            return

        progress_report = self._progress_reports[caller][run_mode][report_num]
        trial_num = self._rich_progress.tasks[progress_report.rich_task_id].completed

        if caller.verbosePref or REPORT_REPORT:
            from pprint import pprint
            pprint(f'{caller.name} {str(context.runmode)} REPORT')

        if self._use_rich:
            if progress_report.num_trials:
                if simulation_mode:
                    num_trials_str = ''
                else:
                    num_trials_str = f' of {progress_report.num_trials}'
            else:
                num_trials_str = ''
            self._rich_progress.update(progress_report.rich_task_id,
                                  description=f'{caller.name}: {run_mode}ed '
                                              f'{trial_num+1}{num_trials_str} trials',
                                  advance=1,
                                  refresh=True)
        if (not simulation_mode
                and progress_report.num_trials
                and (trial_num == progress_report.num_trials)):
            self._progress_reports[caller][run_mode].pop()

    def report_output(self, caller, report_num, scheduler, show_output, content, context, nodes_to_report=False,
                      node=None):

        if report_num is None or show_output is False:
            return
        # if show_report show_output None, defer to Composition's reportOutputPref
        if show_output is None:  # if it is False, leave as is to suppress output
            show_output = caller.reportOutputPref
        if show_output:
            show_output = str(show_output)

        try:
            if TERSE in show_output:   # give precedence to argument in call to execute
                rich_report = False
            else:
                rich_report = True
        except TypeError:
                rich_report = True

        simulation_mode = context.runmode & ContextFlags.SIMULATION_MODE
        if simulation_mode:
            run_mode = SIMULATION
        else:
            run_mode = DEFAULT

        progress_report = self._progress_reports[caller][run_mode][report_num]

        # FIX:  THIS IS A HACK TO FIX THE FACT THAT trial_num SEEMS TO BE DIFFERENT FOR TERSE AND FULL
        trial_num = scheduler.clock.time.trial - (show_output is not TERSE)

        if content is 'trial_init':

            progress_report.trial_report = []

            if show_output is not False:  # if it is False, suppress output
                show_output = show_output or caller.reportOutputPref # if it is None, defer to Composition's
                # reportOutputPref

                #  if rich report, report trial number and Composition's input
                if rich_report:
                    progress_report.trial_report = [f"\n[bold {trial_panel_color}]input:[/]"
                                                     f" {[i.tolist() for i in caller.get_input_values(context)]}"]
                else:
                    # print trial separator and input array to Composition
                    self._rich_progress.console.print(f"[bold {trial_panel_color}]{caller.name} "
                                                      f"TRIAL {trial_num} ====================")

        elif content is 'time_step_init':
            if show_output:
                if rich_report:
                    progress_report.time_step_report = [] # Contains rich.Panel for each node executed in time_step
                elif nodes_to_report:
                    self._rich_progress.console.print(f'[{time_step_panel_color}]'
                                                      f'Time Step {scheduler.clock.time.time_step} ---------')

        elif content is 'node':
            if not node:
                assert False, 'Node not specified in call to PNLProgress report_output'
            if show_output and (node.reportOutputPref or show_output is FULL or show_output is TERSE):
                if rich_report:
                    progress_report.time_step_report.append(
                        self.node_execution_report(node,
                                                   input_val=node.get_input_values(context),
                                                   output_val=node.output_port.parameters.value._get(context),
                                                   show_output=show_output,
                                                   context=context
                                                   ))
                else:
                    self._rich_progress.console.print(f'[{node_panel_color}]{node.name} executed')

        elif content is 'time_step':
            if (show_output and (nodes_to_report or show_output is FULL) and rich_report):
                progress_report.trial_report.append('')
                progress_report.trial_report.append(Panel(RenderGroup(*progress_report.time_step_report),
                                                           # box=box.HEAVY,
                                                           border_style=time_step_panel_color,
                                                           box=time_step_panel_box,
                                                           title=f'[bold {time_step_panel_color}]\nTime Step '
                                                                 f'{scheduler.clock.time.time_step}[/]',
                                                           expand=False))

        elif content is 'trial':
            if show_output and rich_report:
                output_values = []
                for port in caller.output_CIM.output_ports:
                    output_values.append(port.parameters.value._get(context))
                progress_report.trial_report.append(f"\n[bold {trial_output_color}]result:[/]"
                                          f" {[r.tolist() for r in output_values]}\n")
                progress_report.trial_report = Panel(RenderGroup(*progress_report.trial_report),
                                           box=trial_panel_box,
                                           border_style=trial_panel_color,
                                           title=f'[bold{trial_panel_color}] {caller.name}: Trial {trial_num} [/]',
                                           expand=False)

        if content is 'run':
            if show_output and progress_report.trial_report:
                self._rich_progress.console.print(progress_report.trial_report)
                self._rich_progress.console.print('')
                print(self._rich_progress.console.file.getvalue())
                assert True


    @classmethod
    def node_execution_report(cls,
                              node,
                              input_val=None,
                              params=None,
                              output_val=None,
                              show_output=True,
                              context=None):
        from psyneulink.core.components.shellclasses import Function
        from psyneulink.core.globals.keywords import FUNCTION_PARAMS

        node_report = ''

        if show_output is TERSE or node.reportOutputPref is TERSE and show_output is not FULL:
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
