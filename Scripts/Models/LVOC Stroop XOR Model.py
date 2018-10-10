import psyneulink as pnl
import numpy as np

sc = pnl.TransferMechanism(name='Color Stimulus', size=8)
sw = pnl.TransferMechanism(name='Word Stimulus', size=8)

tc = pnl.TransferMechanism(name='Color Task', size=2)
tw = pnl.TransferMechanism(name='Word Task',
                           output_states={pnl.FUNCTION:lambda x: -x})

r = pnl.TransferMechanism(name='Reward')

d = pnl.DDM(name='Task Decision',
            # input_format=pnl.ARRAY,
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
# c.add_projection(sender=tw, receiver=d)
c._analyze_graph()
# c.show_graph()

def o_fct(v):
    return np.sum(v[0]*v[1])
o = pnl.ObjectiveMechanism(monitored_output_states=[d, r],
                           function=o_fct)

lvoc = pnl.LVOCControlMechanism(predictors={pnl.SHADOW_EXTERNAL_INPUTS:[sc,sw]},
                                function=pnl.BayesGLM(mu_0=3),
                                objective_mechanism=o,
                                terminal_objective_mechanism=True,
                                control_signals=[
                                    # '(pnl.SLOPE, tc),
                                    # (pnl.SLOPE, tw)
                                    {'COLOR CONTROL':[(pnl.SLOPE, tc),(pnl.SLOPE, tw)]}
                                ])
c.add_c_node(lvoc)
c._analyze_graph()


c.show_graph()
# input_dict = {m1:[[[1],[1]]],
#               m2:[[1]]}
# input_dict = {m1:[[[1],[1]],[[1],[1]]],
#               m2:[[1],[1]]}
# c.run(inputs=input_dict)
