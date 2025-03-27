import timeit

import numpy as np
from psyneulink import *
import KeysAndDoorsWrapper as kad

# Runtime Switches:
RENDER = False
PNL_COMPILE = False
RUN = True

# *********************************************************************************************************************
# *********************************************** CONSTANTS ***********************************************************
# *********************************************************************************************************************

# temp
obs_len = 7
num_state_nodes = 6

# *********************************************************************************************************************
# **************************************  MECHANISMS AND COMPOSITION  *************************************************
# *********************************************************************************************************************

# Perceptual Mechanism
agent_x = TransferMechanism(name="AGENT X")
agent_y = TransferMechanism(name="AGENT Y")
door_states = TransferMechanism(name="DOOR STATES")
key_states = TransferMechanism(name="KEY STATES")
holding_key = TransferMechanism(name="HOLDING KEY")
key_color = TransferMechanism(name="KEY COLOR")
# Value and Reward Mechanisms (not yet used;  for future use)
actions = TransferMechanism(input_shapes=4, name="ACTIONS")
reward = TransferMechanism(name="REWARD")

# Build EM to this map:
# <     dx = -1
# >     dx = 1
# ^     dy = -1
#   t<<<
#   ^^^^
#   ###^
#   >>>^

em_init_entries = []
num_doors = 0
num_keys = 0
# Translation key
empty = 0
none = 0
false = 0
true = 1
# key colors/door locked colors
red = 1
blue = 2
green = 3
# Additional door states
closed = 4
open = 5
em_init_entries.append(([1], [3], [empty], [empty], [false], [none],
                        [1], [0], [false], [false]))
em_init_entries.append(([2], [3], [empty], [empty], [false], [none],
                        [1], [0], [false], [false]))
em_init_entries.append(([3], [3], [empty], [empty], [false], [none],
                        [0], [-1], [false], [false]))
em_init_entries.append(([3], [2], [empty], [empty], [false], [none],
                        [0], [-1], [false], [false]))
em_init_entries.append(([0], [1], [empty], [empty], [false], [none],
                        [0], [-1], [false], [false]))
em_init_entries.append(([1], [1], [empty], [empty], [false], [none],
                        [0], [-1], [false], [false]))
em_init_entries.append(([2], [1], [empty], [empty], [false], [none],
                        [0], [-1], [false], [false]))
em_init_entries.append(([3], [1], [empty], [empty], [false], [none],
                        [0], [-1], [false], [false]))
# Ending State
em_init_entries.append(([0], [0], [empty], [empty], [false], [none],
                        [0], [0], [false], [false]))
em_init_entries.append(([1], [0], [empty], [empty], [false], [none],
                        [-1], [0], [false], [false]))
em_init_entries.append(([2], [0], [empty], [empty], [false], [none],
                        [-1], [0], [false], [false]))
em_init_entries.append(([3], [0], [empty], [empty], [false], [none],
                        [-1], [0], [false], [false]))
instruct_em = EMComposition(memory_template=em_init_entries, memory_capacity=15,
                            memory_fill=0.001, memory_decay_rate=0,
                            softmax_choice=ARG_MAX,
                            field_weights=(
                                1, 1, 1, 1, 1, 1, None, None, None, None),
                            normalize_memories=False)

def em_ret(state):
    if len(state) == 1:
        return 0, 0, False, False
    instruct_cue = {}
    print(state)
    instruct_cue.update({instruct_em.input_nodes[0]: [[state[0]]]})
    instruct_cue.update({instruct_em.input_nodes[1]: [[state[1]]]})
    instruct_cue.update({instruct_em.input_nodes[2]: [[empty]]})
    instruct_cue.update({instruct_em.input_nodes[3]: [[empty]]})
    instruct_cue.update({instruct_em.input_nodes[4]: [[false]]})
    instruct_cue.update({instruct_em.input_nodes[5]: [[none]]})

    agent_instructions = instruct_em.run(instruct_cue)

    if agent_instructions[8][0] == 0:
        open = False
    else:
        open = True
    if agent_instructions[9][0] == 0:
        pick_up = False
    else:
        pick_up = True
    return np.array([agent_instructions[6][0], agent_instructions[7][0], open, pick_up])

def decision_function(inputs):
    if len(inputs) == 1:
        return em_ret(inputs[0])
    state = [inputs[0], inputs[1]]
    return em_ret(state)

decision_mech = ProcessingMechanism(function=decision_function, name="DECISION MECH")

controlMech = ControlMechanism(
                                name='EM_mech',
                                objective_mechanism=ObjectiveMechanism(monitor=[agent_x, agent_y,
                                                                                door_states, key_states, holding_key,
                                                                                key_color],
                                                                       function=lambda x: np.sum(x)),
                                function=lambda x: np.sum(x)
                               )

output = TransferMechanism(name='output')

# Create Composition
agent_comp = Composition(name='KEYS AND DOORS COMPOSITION')
agent_comp.add_nodes([agent_x, agent_y, door_states, key_states, holding_key, key_color, decision_mech, controlMech,
                      output])

# Connect Nodes
agent_comp.add_projection(MappingProjection(), agent_x, decision_mech)
agent_comp.add_projection(MappingProjection(), agent_y, decision_mech)
agent_comp.add_projection(MappingProjection(), door_states, decision_mech)
agent_comp.add_projection(MappingProjection(), key_states, decision_mech)
agent_comp.add_projection(MappingProjection(), holding_key, decision_mech)
agent_comp.add_projection(MappingProjection(), key_color, decision_mech)
agent_comp.add_projection(MappingProjection(), decision_mech, output)
# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

# Edits will have to be made once the control mechanism is created

num_trials = 1


def main():
    env = kad.KeysAndDoorsEnv(grid="""
                                    t...
                                    ....
                                    ###.
                                    s...
                                    """)
    agent_comp.show_graph()
#     reward = 0
#     done = False
#     print("Running simulation...")
#     steps = 0
#     start_time = timeit.default_timer()
#     for _ in range(num_trials):
#         observation = env.reset()
#         while True:
#             if PNL_COMPILE:
#                 BIN_EXECUTE = 'LLVM'
#             else:
#                 BIN_EXECUTE = 'Python'
#             agent_comp.run(inputs={agent_x: observation[0], agent_y: observation[1],
#                                    door_states: observation[3], key_states: observation[4],
#                                    holding_key: observation[5], key_color: observation[6]},
#                            bin_execute=BIN_EXECUTE
#                            )
#             decision_output = decision_mech.output_values[0]
#             dx, dy, open, pickup = decision_output
#             observation, reward, done = env.step(dx, dy, open, pickup)
#             if RENDER:
#                 env.render()
#             if done:
#                 break
#     stop_time = timeit.default_timer()
#     print(f'{steps / (stop_time - start_time):.1f} steps/second, {steps} total steps in '
#           f'{stop_time - start_time:.2f} seconds')
#
#
if RUN:
    if __name__ == "__main__":
        main()
