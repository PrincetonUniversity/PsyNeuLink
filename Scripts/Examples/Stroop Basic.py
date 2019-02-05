import numpy as np
import psyneulink as pnl

#  INPUT LAYER
import psyneulink.core.components.functions.distributionfunctions
import psyneulink.core.components.functions.transferfunctions

ci = pnl.TransferMechanism(size=2, function=psyneulink.core.components.functions.transferfunctions.Linear, name='COLOR INPUT')
wi = pnl.TransferMechanism(size=2, function=psyneulink.core.components.functions.transferfunctions.Linear, name='WORD INPUT')

# TASK LAYER
t = pnl.TransferMechanism(size=2, function=psyneulink.core.components.functions.transferfunctions.Linear, name='TASK')

#   HIDDEN LAYER
unit_noise = 0.001
processing_rate = 0.1
ch = pnl.TransferMechanism(size=2,
                           function=psyneulink.core.components.functions.transferfunctions.Logistic(gain=1.0, x_0=4.0),  #should be able to get same result with offset = -4.0
                           integrator_mode=False,
                           noise=psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0, standard_deviation=unit_noise).function,
                           integration_rate=processing_rate,
                           name='COLORS HIDDEN')

wh = pnl.TransferMechanism(size=2,
                           function=psyneulink.core.components.functions.transferfunctions.Logistic(gain=1.0, x_0=4.0),
                           integrator_mode=False,
                           noise=psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0, standard_deviation=unit_noise).function,
                           integration_rate=processing_rate,
                           name='WORDS HIDDEN')

# OUTPUT LAYER
r = pnl.TransferMechanism(size=2,
                          function=psyneulink.core.components.functions.transferfunctions.Logistic,
                          integrator_mode=False,
                          noise=psyneulink.core.components.functions.distributionfunctions.NormalDist(mean=0, standard_deviation=unit_noise).function,
                          integration_rate=processing_rate,
                          name='RESPONSE')

# DECISION LAYER
d = pnl.DDM(input_format=pnl.ARRAY)
l = pnl.LCAMechanism(size=2)


#   LOGGING
ch.set_log_conditions('value')
wh.set_log_conditions('value')
r.set_log_conditions('value')

#  PROJECTIONS
c_ih = pnl.MappingProjection(matrix=[[2.2, -2.2], [-2.2, 2.2]], name='COLOR INPUT TO HIDDEN')
w_ih = pnl.MappingProjection(matrix=[[2.6, -2.6], [-2.6, 2.6]], name='WORD INPUT TO HIDDEN')
c_hr = pnl.MappingProjection(matrix=[[1.3, -1.3], [-1.3, 1.3]], name='COLOR HIDDEN TO OUTPUT')
w_hr = pnl.MappingProjection(matrix=[[2.5, -2.5], [-2.5, 2.5]], name='WORD HIDDEN TO OUTPUT')
t_c = pnl.MappingProjection(matrix=[[4.0, 4.0], [0, 0]], name='COLOR NAMING')
t_w = pnl.MappingProjection(matrix=[[0, 0], [4.0, 4.0]], name='WORD READING')

#                   Cong   Incong
inputs_dict = {ci: [[1,0], [1,0]],
               wi: [[1,0], [0,1]],
               t:  [[1,0], [1,0]]}
num_trials = 1

# SYSTEM VERSION ----------------------------------------------------------
#
# PROCESSES
cp = pnl.Process(pathway=[ci, c_ih, ch, c_hr, r], name='COLOR NAMING PROCESS')
wp = pnl.Process(pathway=[wi, w_ih, wh, w_hr, r], name='WORD READING PROCESS')
cnp = pnl.Process(pathway=[t, t_c, ch], name='COLOR NAMING CONTROL PROCESS')
wrp = pnl.Process(pathway=[t, t_w, wh], name='WORD READING CONTROL PROCESS')
rdp = pnl.Process(pathway=[r, d], name='DECISION PROCESS')  # Since projection uses identity matrix, no need to specify
# rdp = pnl.Process(pathway=[r, l], name='DECISION PROCESS') # LCA instead of DDM

# SYSTEM
s = pnl.System(processes=[cp, wp, cnp, wrp, rdp], name='STROOP SYSTEM')
s.show_graph(show_mechanism_structure=True, show_dimensions=pnl.ALL)
print('Results from {} trials of Stroop System:\n'.format(num_trials),
      s.run(num_trials=num_trials, inputs=inputs_dict))


# # COMPOSITION VERSION -------------------------------------------------------
#
# # COMPOSITION
# c = pnl.Composition(name='STROOP COMPOSITION')
# c.add_node(ci)
# c.add_node(wi)
# c.add_node(ch)
# c.add_node(wh)
# c.add_node(t)
# c.add_node(r)
# c.add_node(d)
# c.add_projection(sender=ci, receiver=ch, projection=c_ih)
# c.add_projection(sender=wi, receiver=wh, projection=w_ih)
# c.add_projection(sender=ch, receiver=r, projection=c_hr)
# c.add_projection(sender=wh, receiver=r, projection=w_hr)
# c.add_projection(sender=t, receiver=ch, projection=t_c)
# c.add_projection(sender=t, receiver=ch, projection=t_w)
# c.add_projection(sender=r, receiver=d, projection=pnl.IDENTITY_MATRIX)
# # c.add_projection(sender=r, receiver=l, projection=pnl.IDENTITY_MATRIX)  # LCA instead of DDM

# NOTE: THIS MAY NOT ACCURATELY DEPICT ALL PROJECTIONS (WORK IN PROGRESS!)
# c.show_graph()
# print('Results from {} trials of Stroop Composition:\n'.format(num_trials),
#       c.run(num_trials=num_trials, inputs=inputs_dict))
