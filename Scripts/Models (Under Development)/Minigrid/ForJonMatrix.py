import timeit

import numpy as np
from psyneulink import *
import KeysAndDoorsWrapper as kad

# Runtime Switches:
RENDER = True
PNL_COMPILE = False
RUN = True

# *********************************************************************************************************************
# *********************************************** CONSTANTS ***********************************************************
# *********************************************************************************************************************

# temp
obs_len = 6
num_state_nodes = 6
num_doors = 1
num_keys = 1

# *********************************************************************************************************************
# **************************************  MECHANISMS AND COMPOSITION  *************************************************
# *********************************************************************************************************************

# Perceptual Mechanism
state_input = ProcessingMechanism(name='STATE INPUT',
                                  default_variable=[[0], [0], [0] * num_doors, [0] * num_keys, [0], [0]])
#target_mech = ProcessingMechanism(name='TARGET MECHANISM')

agent_x = TransferMechanism(name="AGENT X")
agent_y = TransferMechanism(name="AGENT Y")
door_states = TransferMechanism(name="DOOR STATES")
key_states = TransferMechanism(name="KEY STATES")
holding_key = TransferMechanism(name="HOLDING KEY")
key_color = TransferMechanism(name="KEY COLOR")
dx = TransferMechanism(name="DX")
dy = TransferMechanism(name="DY")
open_action = TransferMechanism(name="OPEN ACTION")
pickup = TransferMechanism(name="PICKUP")

#output = ProcessingMechanism(name="OUTPUT", default_variable=[[0, 0, 0, 0]])
output_dx = ProcessingMechanism(name="OUTPUT DX")
output_dy = ProcessingMechanism(name="OUTPUT DY")
output_open = ProcessingMechanism(name="OUTPUT OPEN")
output_pickup = ProcessingMechanism(name="OUTPUT PICKUP")

em_init_entries = []
num_doors = 1
num_keys = 1
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


# ORDER:
# X, Y, Door states, Key states, Holding Key, Key Color
# DX, DY, Open, Pickup

instruct_em = EMComposition(name="instruct_em", memory_template=[[0], [0], [0]*num_doors, [0]*num_keys, [0], [0],
                                                                 [0], [0], [0], [0]],
                            memory_capacity=50,
                            memory_decay_rate=0, memory_fill=0.001,
                            fields={"AGENT X": {FIELD_WEIGHT: 1,
                                                LEARN_FIELD_WEIGHT: True,
                                                TARGET_FIELD: False},
                                    "AGENT Y": {FIELD_WEIGHT: 1,
                                                LEARN_FIELD_WEIGHT: True,
                                                TARGET_FIELD: False},
                                    "DOOR STATES": {FIELD_WEIGHT: 1,
                                                    LEARN_FIELD_WEIGHT: True,
                                                    TARGET_FIELD: False},
                                    "KEY STATES": {FIELD_WEIGHT: 1,
                                                   LEARN_FIELD_WEIGHT: True,
                                                   TARGET_FIELD: False},
                                    "HOLDING KEY": {FIELD_WEIGHT: 1,
                                                    LEARN_FIELD_WEIGHT: True,
                                                    TARGET_FIELD: False},
                                    "KEY COLOR": {FIELD_WEIGHT: 1,
                                                  LEARN_FIELD_WEIGHT: True,
                                                  TARGET_FIELD: False},
                                    "DX": {FIELD_WEIGHT: 1,
                                           LEARN_FIELD_WEIGHT: True,
                                           TARGET_FIELD: True},
                                    "DY": {FIELD_WEIGHT: 1,
                                           LEARN_FIELD_WEIGHT: True,
                                           TARGET_FIELD: True},
                                    "OPEN ACTION": {FIELD_WEIGHT: 1,
                                                    LEARN_FIELD_WEIGHT: True,
                                                    TARGET_FIELD: True},
                                    "PICKUP": {FIELD_WEIGHT: 1,
                                               LEARN_FIELD_WEIGHT: True,
                                               TARGET_FIELD: True},
                                    },
                            softmax_choice=WEIGHTED_AVG,
                            normalize_memories=True,
                            enable_learning=True,
                            softmax_gain=1.0)


# Pathways from state to EM
state_to_em_agent_x = [state_input,
                       MappingProjection(matrix=IDENTITY_MATRIX,
                                         sender=state_input.output_ports[0],
                                         receiver=instruct_em.nodes["AGENT X [QUERY]"],
                                         learnable=True),
                       instruct_em
                       ]
state_to_em_agent_y = [state_input,
                       MappingProjection(matrix=IDENTITY_MATRIX,
                                         sender=state_input.output_ports[1],
                                         receiver=instruct_em.nodes["AGENT Y [QUERY]"],
                                         learnable=True),
                       instruct_em
                       ]
state_to_em_door_states = [state_input,
                           MappingProjection(matrix=IDENTITY_MATRIX,
                                             sender=state_input.output_ports[2],
                                             receiver=instruct_em.nodes["DOOR STATES [QUERY]"],
                                             learnable=True),
                           instruct_em
                           ]
state_to_em_key_states = [state_input,
                          MappingProjection(matrix=IDENTITY_MATRIX,
                                            sender=state_input.output_ports[3],
                                            receiver=instruct_em.nodes["KEY STATES [QUERY]"],
                                            learnable=True),
                          instruct_em
                          ]
state_to_em_holding_key = [state_input,
                           MappingProjection(matrix=IDENTITY_MATRIX,
                                             sender=state_input.output_ports[4],
                                             receiver=instruct_em.nodes["HOLDING KEY [QUERY]"],
                                             learnable=True),
                           instruct_em
                           ]
state_to_em_key_color = [state_input,
                         MappingProjection(matrix=IDENTITY_MATRIX,
                                           sender=state_input.output_ports[5],
                                           receiver=instruct_em.nodes["KEY COLOR [QUERY]"],
                                           learnable=True),
                         instruct_em
                         ]

# Pathways from EM to actions
# dx_matrix = np.array([[1, 0, 0, 0]])  # Maps to first position
# dy_matrix = np.array([[0, 1, 0, 0]])  # Maps to second position
# open_matrix = np.array([[0, 0, 1, 0]])  # Maps to third position
# pickup_matrix = np.array([[0, 0, 0, 1]])


state_to_em_dx = [state_input,
                  MappingProjection(matrix=IDENTITY_MATRIX,
                                    sender=state_input,
                                    receiver=instruct_em.nodes["DX [QUERY]"],
                                    learnable=True),
                  instruct_em,
                  MappingProjection(matrix=IDENTITY_MATRIX,
                                    sender=instruct_em.nodes["DX [RETRIEVED]"],
                                    receiver=output_dx,
                                    learnable=False),
                  output_dx
                  ]
state_to_em_dy = [state_input,
                  MappingProjection(matrix=IDENTITY_MATRIX,
                                    sender=state_input,
                                    receiver=instruct_em.nodes["DY [QUERY]"],
                                    learnable=True),
                  instruct_em,
                  MappingProjection(matrix=IDENTITY_MATRIX,
                                    sender=instruct_em.nodes["DY [RETRIEVED]"],
                                    receiver=output_dy,
                                    learnable=False),
                  output_dy
                  ]
state_to_em_open = [state_input,
                    MappingProjection(matrix=IDENTITY_MATRIX,
                                      sender=state_input,
                                      receiver=instruct_em.nodes["OPEN ACTION [QUERY]"],
                                      learnable=True),
                    instruct_em,
                    MappingProjection(matrix=IDENTITY_MATRIX,
                                      sender=instruct_em.nodes["OPEN ACTION [RETRIEVED]"],
                                      receiver=output_open,
                                      learnable=False),
                    output_open
                    ]
state_to_em_pickup = [state_input,
                      MappingProjection(matrix=IDENTITY_MATRIX,
                                        sender=state_input,
                                        receiver=instruct_em.nodes["PICKUP [QUERY]"],
                                        learnable=True),
                      instruct_em,
                      MappingProjection(matrix=IDENTITY_MATRIX,
                                        sender=instruct_em.nodes["PICKUP [RETRIEVED]"],
                                        receiver=output_pickup,
                                        learnable=False),
                      output_pickup
                      ]

# Create Composition
agent_comp = AutodiffComposition([state_to_em_agent_x,
                          state_to_em_agent_y,
                          state_to_em_door_states,
                          state_to_em_key_states,
                          state_to_em_holding_key,
                          state_to_em_key_color,
                          state_to_em_dx,
                          state_to_em_dy,
                          state_to_em_open,
                          state_to_em_pickup],
                         name='KEYS AND DOORS COMPOSITION')


# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

def get_teacher_actions(input_array):
    # Build action if statements to this map:
    # <     dx = -1
    # >     dx = 1
    # ^     dy = -1
    # p     pickup = true
    # .     ignore
    #   t..
    #   ...
    #   ##.
    #   s..
    #
    #   t<<
    #   ^^^
    #   ##^
    #   >>^
    agent_x = input_array[0]
    agent_y = input_array[1]

    if agent_y == 0:
        if agent_x != 0:
            return [-1, 0, 0, 0]
    if agent_y == 1:
        return [0, -1, 0, 0]
    if agent_y == 2:
        return [0, -1, 0, 0]
    if agent_y == 3:
        if agent_x > 1:
            return [0, -1, 0, 0]
        else:
            return [1, 0, 0, 0]

    return [1, 0, 0, 0]



num_trials = 1


def main():
    env = kad.KeysAndDoorsEnv(grid="""
                                    t..
                                    ...
                                    ##.
                                    s..
                                    """)
    total_loss = 0
    reward = 0
    done = False
    print("Running simulation...")
    steps = 0
    start_time = timeit.default_timer()
    #agent_comp.show_graph(show_node_structure=True)
    #agent_comp.infer_backpropagation_learning_pathways(execution_mode=ExecutionMode.PyTorch)
    for _ in range(num_trials):
        observation = env.reset()
        while True:
            if PNL_COMPILE:
                BIN_EXECUTE = 'LLVM'
            else:
                BIN_EXECUTE = 'Python'
            # Format
            input_array = [[observation[0]], [observation[1]], [-1], [-1],
                           [observation[4]], [observation[5]]]
            if num_doors > 1:
                input_array[2] = (x for x in observation[2])
            if num_keys > 1:
                input_array[3] = (x for x in observation[3])

            # Get teacher-forced actions for the current observation
            teacher_actions = get_teacher_actions(input_array)
            # Build the inputs dictionary:
            targets = agent_comp.get_target_nodes()
            inputs = {state_input: input_array,
                      targets[0]: [[teacher_actions[0]]],
                      targets[1]: [[teacher_actions[1]]],
                      targets[2]: [[teacher_actions[2]]],
                      targets[3]: [[teacher_actions[3]]]}

            learning_results = agent_comp.learn(
                inputs=inputs,
                epochs=1,
                learning_rate=0.01
            )

            # Optionally, print the loss if available.
            if learning_results is not None and isinstance(learning_results, (list, tuple)) and len(
                    learning_results) > 0:
                loss = learning_results[0]
                print(f"Loss: {loss}")
                total_loss += loss

            # Run the agent composition using state_input as the input mechanism
            execution = agent_comp.run(
                inputs=inputs
            )

            dx_val = float(instruct_em.nodes["DX [RETRIEVED]"].value[0])
            dy_val = float(instruct_em.nodes["DY [RETRIEVED]"].value[0])
            open_action_val = float(instruct_em.nodes["OPEN ACTION [RETRIEVED]"].value[0])
            pickup_action_val = float(instruct_em.nodes["PICKUP [RETRIEVED]"].value[0])


            observation, reward, done = env.step(dx_val, dy_val, open_action_val, pickup_action_val)

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
