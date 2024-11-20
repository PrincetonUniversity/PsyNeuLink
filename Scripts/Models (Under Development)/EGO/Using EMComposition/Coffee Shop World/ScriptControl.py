from psyneulink.core.compositions.report import ReportOutput, ReportProgress
import psyneulink as pnl

# Settings for running script:

# MODEL_PARAMS = 'TestParams'
MODEL_PARAMS = 'DeclanParams'

CONSTRUCT_MODEL = True                 # THIS MUST BE SET TO True to run the script
DISPLAY_MODEL =  (                     # Only one of the following can be uncommented:
    None                             # suppress display of model
    # {                                  # show simple visual display of model
    #     # 'show_pytorch': True,            # show pytorch graph of model
    #     'show_learning': True,
    #     # 'show_nested_args': {'show_learning': pnl.ALL},
    #     # 'show_projections_not_in_composition': True,
    #     # 'show_nested': {'show_node_structure': True},
    #     # 'exclude_from_gradient_calc_style': 'dashed'# show target mechanisms for learning
    #     # 'show_node_structure': True     # show detailed view of node structures and projections
    # }
)
# RUN_MODEL = False                      # False => don't run the model
RUN_MODEL = True,                       # True => run the model
# REPORT_OUTPUT = ReportOutput.FULL  # Sets console output during run [ReportOutput.ON, .TERSE OR .FULL]
REPORT_OUTPUT = ReportOutput.OFF     # Sets console output during run [ReportOutput.ON, .TERSE OR .FULL]
REPORT_PROGRESS = ReportProgress.OFF # Sets console progress bar during run
PRINT_RESULTS = False                # don't print model.results to console after execution
# PRINT_RESULTS = True                 # print model.results to console after execution
SAVE_RESULTS = False                 # save model.results to disk
# PLOT_RESULTS = False                  # don't plot results (PREDICTIONS) vs. TARGETS
PLOT_RESULTS = True                  # plot results (PREDICTIONS) vs. TARGETS
ANIMATE = False                       # {UNIT:EXECUTION_SET} # Specifies whether to generate animation of execution
