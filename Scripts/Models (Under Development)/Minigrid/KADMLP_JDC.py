import timeit

import numpy as np
import psyneulink
from psyneulink import *
import KeysAndDoorsWrapper as kad
from psyneulink import AutodiffComposition

print(psyneulink.__version__)
# NOTE: The MLP input, while a separate Node than the state input is currently just a workaround to get the input
# to the MLP. For some reason it keeps giving me errors when I try to pass it directly.


# Runtime Switches:
RENDER = True
PNL_COMPILE = False
RUN = True

# *********************************************************************************************************************
# *********************************************** CONSTANTS ***********************************************************
# *********************************************************************************************************************

# temp
obs_len = 8
num_state_nodes = 8
num_doors = 1
num_keys = 1

# *********************************************************************************************************************
# **************************************  MECHANISMS AND COMPOSITION  *************************************************
# *********************************************************************************************************************

# Perceptual Mechanism
state_input = ProcessingMechanism(name='STATE INPUT',
                                  default_variable=[[0], [0], [0] * num_doors, [0] * num_keys, [0], [0], [0], [0]])
agent_x = TransferMechanism(name="AGENT X")
agent_y = TransferMechanism(name="AGENT Y")
door_states = TransferMechanism(name="DOOR STATES")
key_states = TransferMechanism(name="KEY STATES")
holding_key = TransferMechanism(name="HOLDING KEY")
key_color = TransferMechanism(name="KEY COLOR")
heaven = TransferMechanism(name="HEAVEN")
certainty = TransferMechanism(name="CERTAINTY")
right = TransferMechanism(name="RIGHT")
left = TransferMechanism(name="LEFT")
up = TransferMechanism(name="UP")
down = TransferMechanism(name="DOWN")
open_action = TransferMechanism(name="OPEN ACTION")
pickup = TransferMechanism(name="PICKUP")
read = TransferMechanism(name="READ")
output = ProcessingMechanism(name="OUTPUT", default_variable=[[0, 0, 0, 0, 0, 0, 0]])

em_init_entries = []
num_doors = 0
num_keys = 0
# Translation key
empty = -1
none = 0
false = 0
true = 1
# key colors/door locked colors
red = 1
green = 2
blue = 3
# Additional door states
closed = 4
open = 5
# Key state translation
key = 1
no_key = 0
# heaven hell states
h = 1
j = 2
t = 0
# certainty
certain = 1
not_certain = 0

# Build EM to this map:
# <     dx = -1
# >     dx = 1
# ^     dy = -1
# p     pickup = true
# .     ignore

# ORDER:
# X, Y, Door states, Key states, Holding Key, Key Color, Heaven
# DX, DY, Open, Pickup, read
em_init_entries = [
    # Not holding key, door closed, door closed
    ([0], [3], [-1], [-1], [false], [none], [none], [certain],
     [1], [0], [0], [0], [false], [false], [false]),
    ([1], [3], [-1], [-1], [false], [none], [none], [certain],
     [1], [0], [0], [0], [false], [false], [false]),
    ([2], [3], [-1], [-1], [false], [none], [none], [certain],
     [1], [0], [0], [0], [false], [false], [false]),
    ([3], [3], [-1], [-1], [false], [none], [none], [certain],
     [0], [0], [1], [0], [false], [false], [false]),
    ([3], [2], [-1], [-1], [false], [none], [none], [certain],
     [0], [0], [1], [0], [false], [false], [false]),
    ([3], [1], [-1], [-1], [false], [none], [none], [certain],
     [0], [1], [0], [0], [false], [false], [false]),
    ([2], [1], [-1], [-1], [false], [none], [none], [certain],
     [0], [1], [0], [0], [false], [false], [false]),
    ([1], [1], [-1], [-1], [false], [none], [none], [certain],
     [0], [1], [0], [0], [false], [false], [false]),
    ([0], [1], [-1], [-1], [false], [none], [none], [certain],
     [0], [0], [1], [0], [false], [false], [false]),
]
instruct_em = EMComposition(name="instruct_em", memory_template=em_init_entries, memory_capacity=500,
                            memory_decay_rate=0, memory_fill=0.001,
                            fields={"AGENT X": {FIELD_WEIGHT: 1,
                                                LEARN_FIELD_WEIGHT: False,
                                                TARGET_FIELD: False},
                                    "AGENT Y": {FIELD_WEIGHT: 1,
                                                LEARN_FIELD_WEIGHT: False,
                                                TARGET_FIELD: False},
                                    "DOOR STATES": {FIELD_WEIGHT: 1,
                                                    LEARN_FIELD_WEIGHT: False,
                                                    TARGET_FIELD: False},
                                    "KEY STATES": {FIELD_WEIGHT: 1,
                                                   LEARN_FIELD_WEIGHT: False,
                                                   TARGET_FIELD: False},
                                    "HOLDING KEY": {FIELD_WEIGHT: 1,
                                                    LEARN_FIELD_WEIGHT: False,
                                                    TARGET_FIELD: False},
                                    "KEY COLOR": {FIELD_WEIGHT: 1,
                                                  LEARN_FIELD_WEIGHT: False,
                                                  TARGET_FIELD: False},
                                    "HEAVEN": {FIELD_WEIGHT: 1,
                                               LEARN_FIELD_WEIGHT: False,
                                               TARGET_FIELD: False},
                                    "CERTAINTY": {FIELD_WEIGHT: 1,
                                                  LEARN_FIELD_WEIGHT: False,
                                                  TARGET_FIELD: False},
                                    "RIGHT": {FIELD_WEIGHT: None,
                                           LEARN_FIELD_WEIGHT: False,
                                           TARGET_FIELD: True},
                                    "LEFT": {FIELD_WEIGHT: None,
                                           LEARN_FIELD_WEIGHT: False,
                                           TARGET_FIELD: True},
                                    "UP": {FIELD_WEIGHT: None,
                                           LEARN_FIELD_WEIGHT: False,
                                           TARGET_FIELD: True},
                                    "DOWN": {FIELD_WEIGHT: None,
                                           LEARN_FIELD_WEIGHT: False,
                                           TARGET_FIELD: True},
                                    "OPEN ACTION": {FIELD_WEIGHT: None,
                                                    LEARN_FIELD_WEIGHT: False,
                                                    TARGET_FIELD: True},
                                    "PICKUP": {FIELD_WEIGHT: None,
                                               LEARN_FIELD_WEIGHT: False,
                                               TARGET_FIELD: True},
                                    "READ": {FIELD_WEIGHT: None,
                                             LEARN_FIELD_WEIGHT: False,
                                             TARGET_FIELD: True},
                                    },
                            softmax_choice=ARG_MAX,
                            normalize_memories=True,
                            enable_learning=False,
                            softmax_gain=10.0)

# Pathways from state to EM
state_to_em_agent_x = [state_input,
                       MappingProjection(matrix=IDENTITY_MATRIX,
                                         sender=state_input.output_ports[0],
                                         receiver=instruct_em.nodes["AGENT X [QUERY]"],
                                         learnable=False),
                       instruct_em
                       ]
state_to_em_agent_y = [state_input,
                       MappingProjection(matrix=IDENTITY_MATRIX,
                                         sender=state_input.output_ports[1],
                                         receiver=instruct_em.nodes["AGENT Y [QUERY]"],
                                         learnable=False),
                       instruct_em
                       ]
state_to_em_door_states = [state_input,
                           MappingProjection(matrix=IDENTITY_MATRIX,
                                             sender=state_input.output_ports[2],
                                             receiver=instruct_em.nodes["DOOR STATES [QUERY]"],
                                             learnable=False),
                           instruct_em
                           ]
state_to_em_key_states = [state_input,
                          MappingProjection(matrix=IDENTITY_MATRIX,
                                            sender=state_input.output_ports[3],
                                            receiver=instruct_em.nodes["KEY STATES [QUERY]"],
                                            learnable=False),
                          instruct_em
                          ]
state_to_em_holding_key = [state_input,
                           MappingProjection(matrix=IDENTITY_MATRIX,
                                             sender=state_input.output_ports[4],
                                             receiver=instruct_em.nodes["HOLDING KEY [QUERY]"],
                                             learnable=False),
                           instruct_em
                           ]
state_to_em_key_color = [state_input,
                         MappingProjection(matrix=IDENTITY_MATRIX,
                                           sender=state_input.output_ports[5],
                                           receiver=instruct_em.nodes["KEY COLOR [QUERY]"],
                                           learnable=False),
                         instruct_em
                         ]
state_to_em_heaven = [state_input,
                      MappingProjection(matrix=IDENTITY_MATRIX,
                                        sender=state_input.output_ports[6],
                                        receiver=instruct_em.nodes["HEAVEN [QUERY]"],
                                        learnable=False),
                      instruct_em
                      ]
state_to_em_certainty = [state_input,
                         MappingProjection(matrix=IDENTITY_MATRIX,
                                           sender=state_input.output_ports[7],
                                           receiver=instruct_em.nodes["CERTAINTY [QUERY]"],
                                           learnable=False),
                         instruct_em
                         ]

# Pathways from EM to actions
right_matrix = np.array([[1, 0, 0, 0, 0, 0, 0]])  # Maps to first position
left_matrix = np.array([[0, 1, 0, 0, 0, 0, 0]])  # Maps to second position
up_matrix = np.array([[0, 0, 1, 0, 0, 0, 0]])
down_matrix = np.array([[0, 0, 0, 1, 0, 0, 0]])
open_matrix = np.array([[0, 0, 0, 0, 1, 0, 0]])  # Maps to third position
pickup_matrix = np.array([[0, 0, 0, 0, 0, 1, 0]])
read_matrix = np.array([[0, 0, 0, 0, 0, 0, 1]])


state_to_em_right = [state_input,
                  MappingProjection(matrix=IDENTITY_MATRIX,
                                    sender=state_input,
                                    receiver=instruct_em.nodes["RIGHT [VALUE]"],
                                    learnable=False),
                  instruct_em,
                  MappingProjection(matrix=right_matrix,
                                    sender=instruct_em.nodes["RIGHT [RETRIEVED]"],
                                    receiver=output,
                                    learnable=False),
                  output
                  ]
state_to_em_left = [state_input,
                  MappingProjection(matrix=IDENTITY_MATRIX,
                                    sender=state_input,
                                    receiver=instruct_em.nodes["LEFT [VALUE]"],
                                    learnable=False),
                  instruct_em,
                  MappingProjection(matrix=left_matrix,
                                    sender=instruct_em.nodes["LEFT [RETRIEVED]"],
                                    receiver=output,
                                    learnable=False),
                  output
                  ]
state_to_em_up = [state_input,
                  MappingProjection(matrix=IDENTITY_MATRIX,
                                    sender=state_input,
                                    receiver=instruct_em.nodes["UP [VALUE]"],
                                    learnable=False),
                  instruct_em,
                  MappingProjection(matrix=up_matrix,
                                    sender=instruct_em.nodes["UP [RETRIEVED]"],
                                    receiver=output,
                                    learnable=False),
                  output
                  ]
state_to_em_down = [state_input,
                  MappingProjection(matrix=IDENTITY_MATRIX,
                                    sender=state_input,
                                    receiver=instruct_em.nodes["DOWN [VALUE]"],
                                    learnable=False),
                  instruct_em,
                  MappingProjection(matrix=down_matrix,
                                    sender=instruct_em.nodes["DOWN [RETRIEVED]"],
                                    receiver=output,
                                    learnable=False),
                  output
                  ]
state_to_em_open = [state_input,
                    MappingProjection(matrix=IDENTITY_MATRIX,
                                      sender=state_input,
                                      receiver=instruct_em.nodes["OPEN ACTION [VALUE]"],
                                      learnable=False),
                    instruct_em,
                    MappingProjection(matrix=open_matrix,
                                      sender=instruct_em.nodes["OPEN ACTION [RETRIEVED]"],
                                      receiver=output,
                                      learnable=False),
                    output
                    ]
state_to_em_pickup = [state_input,
                      MappingProjection(matrix=IDENTITY_MATRIX,
                                        sender=state_input,
                                        receiver=instruct_em.nodes["PICKUP [VALUE]"],
                                        learnable=False),
                      instruct_em,
                      MappingProjection(matrix=pickup_matrix,
                                        sender=instruct_em.nodes["PICKUP [RETRIEVED]"],
                                        receiver=output,
                                        learnable=False),
                      output
                      ]
state_to_em_read = [state_input,
                    MappingProjection(matrix=IDENTITY_MATRIX,
                                      sender=state_input,
                                      receiver=instruct_em.nodes["READ [VALUE]"],
                                      learnable=False),
                    instruct_em,
                    MappingProjection(matrix=read_matrix,
                                      sender=instruct_em.nodes["READ [RETRIEVED]"],
                                      receiver=output,
                                      learnable=False),
                    output
                    ]

# There were issues with projecting the state input directly through the mlp
# MLP input is a separate input that I send the state input through as a work around
# mlp_input = ProcessingMechanism(name='MLP INPUT', default_variable=[0] * obs_len)

# MLP output mechanism with Softmax activation
# Even with calculating the correct entropy for this it showed no learning when using the logistic function
mlp_output = TransferMechanism(name="MLP OUTPUT",
                               default_variable=[0, 0, 0, 0, 0, 0, 0],
                               function=SoftMax,
                               )

hidden_layer = TransferMechanism(name="HIDDEN",
                                default_variable=[0] * 64,
                                function=ReLU)

learning_rate = 0.01
mlp_pway = [state_input,
            MappingProjection(state_input, hidden_layer,
                              learning_rate = learning_rate),
            hidden_layer,
            MappingProjection(hidden_layer, mlp_output,
                                learning_rate = learning_rate),
            mlp_output]
em_to_mlp_pway = [output,
                  MappingProjection(output, mlp_output, learnable=False),
                  mlp_output]

# Create Composition
agent_comp = AutodiffComposition([state_to_em_agent_x,
                                  state_to_em_agent_y,
                                  state_to_em_door_states,
                                  state_to_em_key_states,
                                  state_to_em_holding_key,
                                  state_to_em_key_color,
                                  state_to_em_heaven,
                                  state_to_em_certainty,
                                  state_to_em_right,
                                  state_to_em_left,
                                  state_to_em_up,
                                  state_to_em_down,
                                  state_to_em_open,
                                  state_to_em_pickup,
                                  state_to_em_read,
                                  mlp_pway,
                                  em_to_mlp_pway],
                                 name='KEYS AND DOORS COMPOSITION')

# print("Calling show_graph for pytorch...")
# agent_comp.show_graph(show_pytorch=True)
print(f"\nLearnable projections in {agent_comp.name} with their learning rates:")
for p in agent_comp.projections:
    if p.learnable:
      print(f"\t{p.name}: {p.learning_rate}")

# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

num_trials = 20


def calculate_entropy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)   # avoid log(0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))
    return entropy / max_entropy

def main():
    env = kad.KeysAndDoorsEnv(grid="""
                                    t...
                                    ....
                                    ###.
                                    s...
                                    """)
    reward = 0
    done = False
    print("Running simulation...")
    steps = 0
    start_time = timeit.default_timer()
    entropy_thresh = 0.5
    # agent_comp.show_graph()
    for _ in range(num_trials):
        observation = env.reset()
        while True:
            if PNL_COMPILE:
                BIN_EXECUTE = 'LLVM'
            else:
                BIN_EXECUTE = 'Python'

            # Format input
            door_input = [observation[2]] if not isinstance(observation[2], list) else observation[2]
            key_input = [observation[3]] if not isinstance(observation[3], list) else observation[3]

            input_array = [[observation[0]], [observation[1]], door_input, key_input,
                           [observation[4]], [observation[5]], [observation[6]], [observation[7]]]
            flattened_input = np.array([observation[0], observation[1],
                                        door_input[0], key_input[0],
                                        observation[4], observation[5],
                                        observation[6], observation[7]])

            # Run the agent composition
            # execution = agent_comp.execute(
            #     inputs={state_input: input_array,
            #             mlp_input: flattened_input},
            #     bin_execute=BIN_EXECUTE
            # )

            # Get outputs - EM output is second to last, MLP output is last
            # em_output_values = execution[-2]
            # mlp_output_vals = execution[-1]
            # entropy = calculate_entropy(mlp_output_vals)
            # print(f"mlp output: {mlp_output_vals}")
            # # Extract EM action values
            # right = float(em_output_values[0])
            # left = float(em_output_values[1])
            # up = float(em_output_values[2])
            # down = float(em_output_values[3])
            # open_action = float(em_output_values[4])
            # pickup_action = float(em_output_values[5])
            # read_action = float(em_output_values[6])
            # em_action = np.array((right, left, up, down, open_action, pickup_action, read_action))

            # Convert MLP logits to action
            # mlp_action = np.zeros(7)
            # mlp_action[np.argmax(mlp_output_vals)] = 1
            #
            # if entropy < entropy_thresh:
            #     print(
            #         f"Step {steps}: Using MLP (entropy: {entropy:.3f}, mlp_action: {mlp_action}, em_action: {em_action})")
            # else:
            #     print(
            #         f"Step {steps}: Using EM (entropy: {entropy:.3f}, mlp_action: {mlp_action}, em_action: {em_action})")

            # train MLP with EM action every step
            # find the actual projections
            hidden_to_output = hidden_layer.efferents[0]
            weights_before = hidden_to_output.matrix.base.copy()
            agent_comp.learn(inputs={state_input: input_array,
                                     # mlp_input: flattened_input
                                     },
                             targets={mlp_output: output.value,
                                      # output: [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
                                      },
                             execution_mode=ExecutionMode.PyTorch
                             )
            weights_after = hidden_to_output.matrix.base.copy()
            weight_change = np.abs(weights_after - weights_before).sum()
            max_gradient = np.abs(weights_after - weights_before).max()
            print(f"Weight change (L1): {weight_change:.6f}")
            print(f"Max gradient: {max_gradient:.6f}")
            print(weight_change)
            print(max_gradient)
            emOutput = output.output_values[-1]
            right = emOutput[0]
            left = emOutput[1]
            up = emOutput[2]
            down = emOutput[3]
            open_action = emOutput[4]
            pickup_action = emOutput[5]
            read_action = emOutput[6]
            # Execute action in environment (always use EM action during training)
            observation, reward, done = env.step(right, left, up, down, open_action, pickup_action, read_action)
            steps += 1

            if RENDER:
                env.render()
            if done:
                break

    stop_time = timeit.default_timer()
    print(f'{steps / (stop_time - start_time):.1f} steps/second, {steps} total steps in '
          f'{stop_time - start_time:.2f} seconds')


if RUN:
    if __name__ == "__main__":
        main()