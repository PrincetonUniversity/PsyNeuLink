import functools
import numpy as np
import psyneulink as pnl
import psyneulink.core.components.functions.transferfunctions


def my_linear_fct(x,
                  m=2.0,
                  b=0.0,
                  params={pnl.ADDITIVE_PARAM:'b',
                          pnl.MULTIPLICATIVE_PARAM:'m'}):
    return m * x + b

def my_simple_linear_fct(x,
                         m=1.0,
                         b=0.0
                         ):
    return m * x + b

def my_exp_fct(x,
               r=1.0,
               # b=pnl.CONTROL,
               b=0.0,
               params={pnl.ADDITIVE_PARAM:'b',
                       pnl.MULTIPLICATIVE_PARAM:'r'}
               ):
    return x**r + b

def my_sinusoidal_fct(input,
                      phase=0,
                      amplitude=1,
                      params={pnl.ADDITIVE_PARAM:'phase',
                              pnl.MULTIPLICATIVE_PARAM:'amplitude'}):
    frequency = input[0]
    t = input[1]
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

Input_Layer = pnl.TransferMechanism(
    name='Input_Layer',
    default_variable=np.zeros((2,)),
    function=psyneulink.core.components.functions.transferfunctions.Logistic
)

Output_Layer = pnl.TransferMechanism(
        name='Output_Layer',
        default_variable=[0, 0, 0],
        function=psyneulink.core.components.functions.transferfunctions.Linear,
        # function=pnl.Logistic,
        # output_ports={pnl.NAME: 'RESULTS USING UDF',
        #                pnl.VARIABLE: [(pnl.OWNER_VALUE,0), pnl.TIME_STEP],
        #                pnl.FUNCTION: my_sinusoidal_fct}
        output_ports={pnl.NAME: 'RESULTS USING UDF',
                       # pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
                       pnl.FUNCTION: psyneulink.core.components.functions.transferfunctions.Linear(slope=pnl.GATING)
                       # pnl.FUNCTION: pnl.Logistic(gain=pnl.GATING)
                       # pnl.FUNCTION: my_linear_fct
                       # pnl.FUNCTION: my_exp_fct
                       # pnl.FUNCTION:pnl.UserDefinedFunction(custom_function=my_simple_linear_fct,
                       #                                      params={pnl.ADDITIVE_PARAM:'b',
                       #                                              pnl.MULTIPLICATIVE_PARAM:'m',
                       #                                              },
                                                            # m=pnl.GATING,
                                                            # b=2.0
                                                            # )
                       }
)

Gating_Mechanism = pnl.GatingMechanism(
    # default_gating_allocation=0.0,
    size=[1],
    gating_signals=[
        # Output_Layer
        Output_Layer.output_port,
    ]
)


def print_header(system):
    print("\n\n**** Time: ", system.scheduler.get_clock(system).simple_time)


def show_target(context=None):
    print('Gated: ',
          Gating_Mechanism.gating_signals[0].efferents[0].receiver.owner.name,
          Gating_Mechanism.gating_signals[0].efferents[0].receiver.name)
    print('- Input_Layer.value:                  ', Input_Layer.parameters.value.get(context))
    print('- Output_Layer.value:                 ', Output_Layer.parameters.value.get(context))
    print('- Output_Layer.output_port.variable: ', Output_Layer.output_port.parameters.variable.get(context))
    print('- Output_Layer.output_port.value:    ', Output_Layer.output_port.parameters.value.get(context))

stim_list = {
    Input_Layer: [[-1, 30], [-1, 30], [-1, 30], [-1, 30]],
    Gating_Mechanism: [[0.0], [0.5], [1.0], [2.0]]
}
comp = pnl.Composition(pathways=[[Input_Layer, Output_Layer],[Gating_Mechanism]])
comp.show_graph()
comp.run(num_trials=4,
         inputs=stim_list,
         call_before_trial=functools.partial(print_header, comp),
         call_after_trial=show_target,
)
