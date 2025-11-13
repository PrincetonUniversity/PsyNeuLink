

import psyneulink as pnl


DISPLAY_MODEL = False

#region   MODEL
# ======================================================================================================================
#                                                      MODEL
# ======================================================================================================================

def construct_model(
        state_size
)->pnl.Composition:


    state_input_layer = pnl.ProcessingMechanism(
        name='STATE',
        input_shapes=state_size
    )


    em = pnl.EMComposition(name='em',
                       memory_template=[[0] * state_size],   # state
                    memory_decay_rate=0,
                       memory_fill=0.01,
                       memory_capacity=10,
                       softmax_gain=1.,
                       # Input Nodes:
                       field_names=['STATE'],
                       field_weights=(1))

    EGO_comp = pnl.Composition(name='ego_revaluation')

    # Nodes not included in (decision output) Pathway specified above
    EGO_comp.add_nodes([state_input_layer,
                        em
                        ])

    # Projections:
    QUERY = ' [QUERY]'
    VALUE = ' [VALUE]'
    RETRIEVED = ' [RETRIEVED]'

    # EM encoding --------------------------------------------------------------------------------
    # state -> em
    EGO_comp.add_projection(
        pnl.MappingProjection(state_input_layer, em.nodes['STATE' + QUERY]))
    # time -> em


    return EGO_comp, state_input_layer

#region SCRIPT EXECUTION
# ======================================================================================================================
#                                                   SCRIPT EXECUTION
# ======================================================================================================================

if __name__ == '__main__':

    model, state_input_layer = construct_model(2)

    if DISPLAY_MODEL:
        model.show_graph()


    states = [
        [0, 1],
        [1, 0],
        [0, 1],
    ]


    inputs = {
        state_input_layer: states
    }

    model.run(inputs)

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
