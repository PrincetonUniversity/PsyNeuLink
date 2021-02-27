import time, re, sys, warnings

from psyneulink.core.globals.keywords import FULL
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.utilities import convert_to_list

from rich import print, box
from rich.console import RenderGroup
from rich.progress import Progress as RichProgress
from rich.panel import Panel


class PNLProgressError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ProgressReport():
    def __init__(self, id, runmode, num_trials):
        self.runmode = runmode  # indicates whether run is in DEFAULT_MODE or SIMULATION_MODE
        self.num_trials = num_trials
        self.progress_report_id = id # used for task id in rich
        self.trial_report = []
        self.time_step_report = []


class PNLProgress:
    """
    A singleton context object that provides interface to rich progress bars and pnl_view.
    It returns the currently active progress context instance if one has been instantiated already in another scope.
    It deallocates the progress bar when the outermost context is released.
    """
    _instance = None

    def __new__(cls, show_progress=True) -> 'PNLProgress':
        if cls._instance is None:
            cls._instance = super(PNLProgress, cls).__new__(cls)

            show_progress = convert_to_list(show_progress)

            cls._use_rich = False not in show_progress and (k in show_progress for k in {True, 'rich'})
            cls._use_pnl_view = False not in show_progress and (k in show_progress for k in {True, 'pnl_view'})

            # Check for specification of rich
            if cls._use_rich:
                # Instantiate a rich progress context\object
                # - it is not started until the self.start_progress_report() method is called
                # - auto_refresh is disabled to accomodate IDEs (such as PyCharm and Jupyter Notebooks
                # cls._instance._rich_progress = RichProgress(disable=show_rich, auto_refresh=False)
                cls._instance._rich_progress = RichProgress(auto_refresh=False)

            # Check for specification of pnl_view
            if cls._use_pnl_view:
                warnings.warn("'pnl_view' not yet supported as an option for show_progress of Composition.run()")

            cls._progress_reports = []

            # This counter is incremented on each context __enter__ and decrements
            # on each __exit__. We need this to make sure we don't call progress
            # stop until all references have been released.
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

        if self._use_rich:

            # Simulation mode:
            if context.runmode & ContextFlags.SIMULATION_MODE:
                run_mode = 'Simulat'
                visible = False
            else:
                run_mode = 'Execut'
                visible = True

            # when num_trials is not known (e.g., a generator is for inputs)
            # FIX: NEED TO ADD _start SOMEWHERE
            if num_trials == sys.maxsize:
                start = False
                num_trials = 0
            else:
                start = True

            # FIX: CONTEXTUALIZE FOR RICH
            id = self._rich_progress.add_task(f"[red]{run_mode}ing {comp.name}...",
                                         total=num_trials,
                                         start=start,
                                         visible=visible
                                         )
            return ProgressReport(id, run_mode, num_trials)

            # FIX: ??KEEP:

    def report_progress(self, caller, progress_report, trial_num):
        if self._use_rich:
            if isinstance(trial_num, int):
                if progress_report.num_trials:
                    num_trials_str = f' of {progress_report.num_trials}'
                else:
                    num_trials_str = ''
                self._rich_progress.update(progress_report.progress_report_id,
                                      description=f'{caller.name}: '
                                                  f'{progress_report.runmode}ed {trial_num+1}{num_trials_str} trials',
                                      advance=1,
                                      refresh=True)
            else:
                assert False, f"Invalid 'trial_num' arg to PNLProgress.report_progress from {caller.name}: " \
                              f"'{trial_num}'"

    def report_output(self, caller, progress_report, scheduler, show_output, content, context, nodes_to_report=False,
                      node=None):

        if not progress_report:
            return

        # if it is None, defer to Composition's # reportOutputPref
        if show_output is not False:  # if it is False, leave as is to suppress output
            show_output = show_output or caller.reportOutputPref
        if show_output:
            show_output = str(show_output)

        try:
            if 'terse' in show_output:   # give precedence to argument in call to execute
                rich_report = False
            # elif 'terse' in self.reportOutputPref:
            #     rich_report = False
            else:
                rich_report = True
        except TypeError:
                rich_report = True

        trial_num = scheduler.clock.time.trial

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
                    print(f"[bold {trial_panel_color}]{caller.name} TRIAL {trial_num} ====================")

        elif content is 'time_step_init':
            if show_output:
                if rich_report:
                    progress_report.time_step_report = [] # Contains rich.Panel for each node executed in time_step
                elif nodes_to_report:
                    print(f'[{time_step_panel_color}]Time Step {scheduler.clock.time.time_step} ---------')

        elif content is 'node':
            if not node:
                assert False  # FIX: NEED ERROR MESSAGE HERE
            if show_output and (node.reportOutputPref or show_output is FULL):
                if rich_report:
                    progress_report.time_step_report.append(
                        _report_node_execution(node,
                                               input_val=node.get_input_values(context),
                                               output_val=node.output_port.parameters.value._get(context),
                                               context=context
                                               ))
                else:
                    print(f'[{node_panel_color}]{node.name} executed')

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


# ####################################################
# # An Example
# ####################################################
#
#
# def another_run(task_num):
#     """A simple function to generate another task progress bar."""
#     with PNLProgress() as progress:
#         task = progress.add_task(f"[white]Another Task {task_num} ...", total=100)
#
#         for i in range(100):
#             progress.update(task, advance=1)
#             time.sleep(0.001)
#
#         # We can remove a task if we want to. I notice rich gets pretty slow and can even cause the terminal to hang
#         # if a large number of tasks are created so you might need to do this. It seems to get bad around 100 or so
#         # tasks on my machine.
#         progress.remove_task(task)
#
#
# def run(show_progress: bool = True):
#
#     with PNLProgress(disable=not show_progress) as progress:
#
#         task1 = progress.add_task("[red]Downloading...", total=100)
#         task2 = progress.add_task("[green]Processing...", total=100)
#         task3 = progress.add_task("[cyan]Cooking...", total=100)
#
#         i = 0
#         sub_task_num = 0
#
#         while not progress.finished:
#             progress.update(task1, advance=0.5)
#             progress.update(task2, advance=0.3)
#             progress.update(task3, advance=0.9)
#             time.sleep(0.002)
#
#             # Run another whole task every 30 iterations
#             if i % 30 == 0:
#                 sub_task_num = sub_task_num + 1
#                 another_run(sub_task_num)
#
#             i = i + 1
#
# run()
#
# print("Run Again ... Progress Disabled")
# run(show_progress=False)
#
# print("Run Again")
# run()



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
# ---------------------------------------------------

def _report_node_execution(node,
                           input_val=None,
                           params=None,
                           output_val=None,
                           context=None):
        from psyneulink.core.components.shellclasses import Function
        from psyneulink.core.globals.keywords import FUNCTION_PARAMS

        node_report = ''

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

