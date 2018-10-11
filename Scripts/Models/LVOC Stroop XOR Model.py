import psyneulink as pnl
import numpy as np

def w_fct(stim, color_control):
    '''function for tw, to modulate strength of word reading based on 1 - strength of color naming ControlSignal'''
    return stim * (1-color_control)
w_fct_UDF = pnl.UserDefinedFunction(custom_function=w_fct, color_control=1)


def objective_function(v):
    '''function used for ObjectiveMechanism of lvoc
     v[0] = output of DDM: [probability of color naming, probability of word reading]
     v[1] = reward:        [color naming rewarded, word reading rewarded]
     '''
    return np.sum(v[0]*v[1])


sc = pnl.TransferMechanism(name='Color Stimulus', size=8)
sw = pnl.TransferMechanism(name='Word Stimulus', size=8)

tc = pnl.TransferMechanism(name='Color Task')
tw = pnl.ProcessingMechanism(name='Word Task', function=w_fct_UDF)

r = pnl.TransferMechanism(name='Reward', size=2)

d = pnl.DDM(name='Task Decision',
            # function=pnl.NavarroAndFuss,
            output_states=[pnl.DDM_OUTPUT.PROBABILITY_UPPER_THRESHOLD,
                           pnl.DDM_OUTPUT.PROBABILITY_LOWER_THRESHOLD])

c = pnl.Composition(name='Stroop XOR Model')
c.add_c_node(sc, required_roles=pnl.CNodeRole.ORIGIN)
c.add_c_node(sw, required_roles=pnl.CNodeRole.ORIGIN)
c.add_c_node(tc, required_roles=pnl.CNodeRole.ORIGIN)
c.add_c_node(tw, required_roles=pnl.CNodeRole.ORIGIN)
c.add_c_node(r, required_roles=pnl.CNodeRole.ORIGIN)
c.add_c_node(d, required_roles=pnl.CNodeRole.ORIGIN)
c.add_projection(sender=tc, receiver=d)
c.add_projection(sender=tw, receiver=d)

lvoc = pnl.LVOCControlMechanism(predictors={pnl.SHADOW_EXTERNAL_INPUTS:[sc,sw]},
                                function=pnl.BayesGLM(mu_0=3),
                                objective_mechanism=pnl.ObjectiveMechanism(monitored_output_states=[d, r],
                                                                           function=objective_function),
                                terminal_objective_mechanism=True,
                                control_signals=[{'COLOR CONTROL':[(pnl.SLOPE, tc),
                                                                   ('color_control', tw)]}])
c.add_c_node(lvoc)
c._analyze_graph()


c.show_graph()
# input_dict = {m1:[[[1],[1]]],
#               m2:[[1]]}
# input_dict = {m1:[[[1],[1]],[[1],[1]]],
#               m2:[[1],[1]]}
# c.run(inputs=input_dict)

