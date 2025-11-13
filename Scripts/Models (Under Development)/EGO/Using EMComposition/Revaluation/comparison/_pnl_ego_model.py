import numpy as np

import psyneulink as pnl

import comparison.params as params
import comparison.data as data

DISPLAY_MODEL = False


# region   MODEL
# ======================================================================================================================
#                                                      MODEL
# ======================================================================================================================

def construct_model() -> pnl.Composition:
    state_input_name = 'STATE'
    time_input_name = 'TIME'
    reward_input_name = 'REWARD'
    context_name = 'CONTEXT'

    state_input_layer = pnl.ProcessingMechanism(
        name=state_input_name,
        input_shapes=params.STATE_SIZE)
    time_input_layer = pnl.ProcessingMechanism(
        name=time_input_name,
        input_shapes=params.TIME_SIZE)
    reward_input_layer = pnl.ProcessingMechanism(
        name=reward_input_name,
        input_shapes=params.REWARD_SIZE)
    context_layer = pnl.RecurrentTransferMechanism(
        name=context_name,
        input_shapes=params.CONTEXT_SIZE,
        auto=params.OLD_CONTEXT_INTEGRATION_RATE,
        hetero=0.0)

    em = pnl.EMComposition(name='em',
                           memory_template=[[0] * params.STATE_SIZE,  # state
                                            [0] * params.TIME_SIZE,  # time
                                            [0] * params.CONTEXT_SIZE,  # context
                                            [0] * params.REWARD_SIZE],  # reward
                           memory_fill=params.MEMORY_INITIALIZATION,
                           memory_capacity=params.N_EXPERIENCE_SEQS,
                           memory_decay_rate=0,
                           softmax_gain=1.0 / params.TEMPERATURE,
                           # softmax_threshold=0.001,
                           fields={
                               state_input_name: {
                                   pnl.FIELD_WEIGHT: params.STATE_RETRIEVAL_WEIGHT,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False
                               },
                               time_input_name: {
                                   pnl.FIELD_WEIGHT: params.TIME_RETRIEVAL_WEIGHT,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False},
                               context_name: {
                                   pnl.FIELD_WEIGHT: params.CONTEXT_RETRIEVAL_WEIGHT,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False},
                               reward_input_name: {
                                   pnl.FIELD_WEIGHT: params.REWARD_RETRIEVAL_WEIGHT,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False}
                           })

    EGO_comp = pnl.Composition(name='ego_revaluation')

    # Nodes not included in (decision output) Pathway specified above
    EGO_comp.add_nodes([state_input_layer,
                        time_input_layer,
                        context_layer,
                        reward_input_layer,
                        em
                        ])

    # Projections:
    QUERY = ' [QUERY]'
    VALUE = ' [VALUE]'
    RETRIEVED = ' [RETRIEVED]'

    # EM encoding --------------------------------------------------------------------------------
    # state -> em
    EGO_comp.add_projection(
        pnl.MappingProjection(state_input_layer, em.nodes[state_input_name + QUERY]))
    # time -> em
    EGO_comp.add_projection(
        pnl.MappingProjection(time_input_layer, em.nodes[time_input_name + QUERY]))
    # context -> em
    EGO_comp.add_projection(
        pnl.MappingProjection(context_layer, em.nodes[context_name + QUERY]))
    # reward -> em
    EGO_comp.add_projection(
        pnl.MappingProjection(reward_input_layer, em.nodes[reward_input_name + VALUE]))

    # Inputs to Context ---------------------------------------------------------------------------
    # retrieved context -> context_layer
    EGO_comp.add_projection(
        pnl.MappingProjection(em.nodes[context_name + RETRIEVED], context_layer,
                              matrix=np.eye(params.CONTEXT_SIZE) * params.RETRIEVED_CONTEXT_INTEGRATION_RATE))
    # state -> context_layer
    EGO_comp.add_projection(
        pnl.MappingProjection(state_input_layer, context_layer,
                              matrix=np.eye(params.CONTEXT_SIZE) * params.STATE_INTEGRATION_RATE))

    return EGO_comp, state_input_layer, time_input_layer, reward_input_layer


# region SCRIPT EXECUTION
# ======================================================================================================================
#                                                   SCRIPT EXECUTION
# ======================================================================================================================

if __name__ == '__main__':
    model, state_input_layer, time_input_layer, reward_input_layer = construct_model()

    if DISPLAY_MODEL:
        model.show_graph()

    states, rewards = data.get_baseline_trials(num_seqs=10)
    times = data.get_time_sequence(num_trials=len(states))

    states = np.array(states)
    times = np.array(times)
    inputs = {
        state_input_layer: states,
        time_input_layer: times,
        reward_input_layer: rewards,
    }

    model.run(inputs=inputs)

    print(model.results)

    # if RUN_MODEL:
    #     experience_inputs = build_experience_inputs(state_size=STATE_SIZE,
    #                                                 time_drift_rate=TIME_DRIFT_RATE,
    #                                                 num_baseline_seqs=NUM_BASELINE_SEQS,
    #                                                 num_revaluation_seqs=NUM_REVALUATION_SEQS,
    #                                                 reward_vals=REWARD_VALS,
    #                                                 sampling_type=SAMPLING_TYPE,
    #                                                 ratio=RATIO,
    #                                                 stim_seqs=STIM_SEQS)
    #
    #     prediction_inputs = build_prediction_inputs(state_size=STATE_SIZE,
    #                                                 time_drift_rate=TIME_DRIFT_RATE,
    #                                                 num_roll_outs_per_stim=int(NUM_ROLL_OUTS / 2),
    #                                                 stim_seqs=STIM_SEQS,
    #                                                 reward_vals=REWARD_VALS,
    #                                                 seq_type=PREDICT_SEQ_TYPE)
    #
    #     print(experience_inputs)
    #
    #     input_layers = [TIME_INPUT_LAYER_NAME,
    #                     TASK_INPUT_LAYER_NAME,
    #                     STATE_INPUT_LAYER_NAME,
    #                     REWARD_INPUT_LAYER_NAME]
    #
    #     # Experience Phase
    #     print(f"Presenting {model.name} with {TOTAL_NUM_EXPERIENCE_STIMS} EXPERIENCE stimuli")
    #     model.run(inputs={k: v for k, v in zip(input_layers, experience_inputs)},
    #               execution_mode=EXECUTION_MODE,
    #               report_output=REPORT_OUTPUT,
    #               report_progress=REPORT_PROGRESS)
    #
    #
    #
    #     # Prediction Phase
    #
    #     print(f"Running {model.name} for {NUM_ROLL_OUTS} PREDICT (ROLL OUT) trials")
    #     model.termination_processing = {
    #         TimeScale.TRIAL: And(Condition(lambda: model.nodes[TASK_INPUT_LAYER_NAME].value == Task.PREDICT),
    #                              Condition(lambda: model.nodes[RETRIEVED_REWARD_NAME].value),
    #                              # JustRan(model.nodes[DECISION_LAYER_NAME])
    #                              AllHaveRun()
    #                              )
    #     }
    #     model.run(inputs={k: v for k, v in zip(input_layers, prediction_inputs)},
    #               report_output=REPORT_OUTPUT,
    #               report_progress=REPORT_PROGRESS
    #               )
    #
    #     if PRINT_RESULTS:
    #         print(f"Predicted reward for last stimulus: {model.results}")
