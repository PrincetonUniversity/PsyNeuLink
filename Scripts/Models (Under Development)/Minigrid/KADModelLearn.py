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

num_doors = 1
num_keys = 1

# *********************************************************************************************************************
# **************************************  MECHANISMS AND COMPOSITION  *************************************************
# *********************************************************************************************************************

# Perceptual Mechanism produces 6 values:
# [agent_x, agent_y, door_states, key_states, holding_key, key_color]
state_input = ProcessingMechanism(
    name='STATE INPUT',
    default_variable=[[0], [0], [0] * num_doors, [0] * num_keys, [0], [0]]
)

# Create the EMComposition with a memory template of 10 elements:
# The first 6 fields (query) will be set from state. the last 4 (actions) are computed by the EM.
instruct_em = EMComposition(
    name="instruct_em",
    memory_template=[[0], [0], [0] * num_doors, [0] * num_keys, [0], [0],
                     [0], [0], [0], [0]],
    memory_capacity=50,
    memory_decay_rate=0,
    memory_fill=0.001,
    fields={
        "AGENT X":      {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: True, TARGET_FIELD: False},
        "AGENT Y":      {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: True, TARGET_FIELD: False},
        "DOOR STATES":  {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: True, TARGET_FIELD: False},
        "KEY STATES":   {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: True, TARGET_FIELD: False},
        "HOLDING KEY":  {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: True, TARGET_FIELD: False},
        "KEY COLOR":    {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: True, TARGET_FIELD: False},
        "DX":           {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: True, TARGET_FIELD: True},
        "DY":           {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: True, TARGET_FIELD: True},
        "OPEN ACTION":  {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: True, TARGET_FIELD: True},
        "PICKUP":       {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: True, TARGET_FIELD: True},
    },
    softmax_choice=WEIGHTED_AVG,
    normalize_memories=True,
    enable_learning=True,
    softmax_gain=1.0
)

# ------------------------------------------------------------------------------
proj_agent_x = MappingProjection(
    matrix=[[1]],
    sender=state_input.output_ports[0],
    receiver=instruct_em.nodes["AGENT X [QUERY]"],
    learnable=True
)
proj_agent_y = MappingProjection(
    matrix=[[1]],
    sender=state_input.output_ports[1],
    receiver=instruct_em.nodes["AGENT Y [QUERY]"],
    learnable=True
)
proj_door_states = MappingProjection(
    matrix=[[1]],
    sender=state_input.output_ports[2],
    receiver=instruct_em.nodes["DOOR STATES [QUERY]"],
    learnable=True
)
proj_key_states = MappingProjection(
    matrix=[[1]],
    sender=state_input.output_ports[3],
    receiver=instruct_em.nodes["KEY STATES [QUERY]"],
    learnable=True
)
proj_holding = MappingProjection(
    matrix=[[1]],
    sender=state_input.output_ports[4],
    receiver=instruct_em.nodes["HOLDING KEY [QUERY]"],
    learnable=True
)
proj_color = MappingProjection(
    matrix=[[1]],
    sender=state_input.output_ports[5],
    receiver=instruct_em.nodes["KEY COLOR [QUERY]"],
    learnable=True
)

# ------------------------------------------------------------------------------
pathway_dx = [state_input,
              instruct_em,
              instruct_em.nodes["DX [RETRIEVED]"]]
pathway_dy = [state_input,
              instruct_em,
              instruct_em.nodes["DY [RETRIEVED]"]]
pathway_open = [state_input,
                instruct_em,
                instruct_em.nodes["OPEN ACTION [RETRIEVED]"]]
pathway_pickup = [state_input,
                  instruct_em,
                  instruct_em.nodes["PICKUP [RETRIEVED]"]]

# ------------------------------------------------------------------------------
# Create the autodiff composition using the four pathways:
agent_comp = AutodiffComposition([pathway_dx, pathway_dy, pathway_open, pathway_pickup],
                                 name='KEYS AND DOORS COMPOSITION')

# Add the retrieved action nodes as target nodes for learning.
agent_comp.add_node(instruct_em.nodes["DX [RETRIEVED]"])
agent_comp.add_node(instruct_em.nodes["DY [RETRIEVED]"])
agent_comp.add_node(instruct_em.nodes["OPEN ACTION [RETRIEVED]"])
agent_comp.add_node(instruct_em.nodes["PICKUP [RETRIEVED]"])

agent_comp.nodes_to_roles[instruct_em.nodes["DX [RETRIEVED]"]] = {NodeRole.TARGET, NodeRole.LEARNING}
agent_comp.nodes_to_roles[instruct_em.nodes["DY [RETRIEVED]"]] = {NodeRole.TARGET, NodeRole.LEARNING}
agent_comp.nodes_to_roles[instruct_em.nodes["OPEN ACTION [RETRIEVED]"]] = {NodeRole.TARGET, NodeRole.LEARNING}
agent_comp.nodes_to_roles[instruct_em.nodes["PICKUP [RETRIEVED]"]] = {NodeRole.TARGET, NodeRole.LEARNING}

# Mark the query nodes for learning so that gradients flow back:
agent_comp.nodes_to_roles[instruct_em.nodes["AGENT X [QUERY]"]] = {NodeRole.LEARNING}
agent_comp.nodes_to_roles[instruct_em.nodes["AGENT Y [QUERY]"]] = {NodeRole.LEARNING}
agent_comp.nodes_to_roles[instruct_em.nodes["DOOR STATES [QUERY]"]] = {NodeRole.LEARNING}
agent_comp.nodes_to_roles[instruct_em.nodes["KEY STATES [QUERY]"]] = {NodeRole.LEARNING}
agent_comp.nodes_to_roles[instruct_em.nodes["HOLDING KEY [QUERY]"]] = {NodeRole.LEARNING}
agent_comp.nodes_to_roles[instruct_em.nodes["KEY COLOR [QUERY]"]] = {NodeRole.LEARNING}

# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

def get_teacher_actions(input_array):
    # Use the first two values (agent_x and agent_y) to decide the action.
    agent_x_val = input_array[0]
    agent_y_val = input_array[1]
    if agent_y_val == 0:
        if agent_x_val != 0:
            return [-1, 0, 0, 0]
    if agent_y_val == 1:
        return [0, -1, 0, 0]
    if agent_y_val == 2:
        return [0, -1, 0, 0]
    if agent_y_val == 3:
        if agent_x_val > 1:
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
    # Uncomment to view the composition graph:
    #agent_comp.show_graph()
    agent_comp.infer_backpropagation_learning_pathways(execution_mode=ExecutionMode.PyTorch)
    for _ in range(num_trials):
        observation = env.reset()
        while True:
            BIN_EXECUTE = 'LLVM' if PNL_COMPILE else 'Python'
            # Format input: a 6-element list.
            input_array = [
                [observation[0]],   # agent_x
                [observation[1]],   # agent_y
                [-1],               # door_states
                [-1],               # key_states
                [observation[4]],   # holding_key
                [observation[5]]    # key_color
            ]
            if num_doors > 1:
                input_array[2] = (x for x in observation[2])
            if num_keys > 1:
                input_array[3] = (x for x in observation[3])
            # Compute teacher targets (for DX, DY, OPEN ACTION, PICKUP) from the observation.
            teacher_actions = get_teacher_actions(input_array)
            inputs = {state_input: input_array}

            targets = agent_comp.get_targets()
            teacher_targets = {
                targets[0]: [[teacher_actions[0]]],
                targets[1]: [[teacher_actions[1]]],
                targets[2]: [[teacher_actions[2]]],
                targets[3]: [[teacher_actions[3]]]
            }

            learning_results = agent_comp.learn(
                inputs=inputs,
                targets=teacher_targets,
                epochs=1,
                learning_rate=0.01,
                execution_mode=ExecutionMode.PyTorch
            )

            if learning_results is not None and isinstance(learning_results, (list, tuple)) and len(learning_results) > 0:
                loss = learning_results[0]
                print(f"Loss: {loss}")
                total_loss += loss

            # Run the composition:
            execution = agent_comp.run(inputs=inputs)

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
    print(f'{steps / (stop_time - start_time):.1f} steps/second, {steps} total steps in {stop_time - start_time:.2f} seconds')

if RUN:
    if __name__ == "__main__":
        main()
