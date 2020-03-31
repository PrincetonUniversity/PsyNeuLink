import numpy as np
import os
import psyneulink as pnl
import pytest

# stroop stimuli
red = [1, 0]
green = [0, 1]
word = [0, 1]
color = [1, 0]

num_trials = 5
stroop_stimuli = {
    'color_input': [red] * num_trials,
    'word_input': [green] * num_trials,
    'task_input': [color] * num_trials,
}


@pytest.mark.parametrize(
    'filename, composition_name, input_dict_str',
    [
        ('model_basic.py', 'comp', '{A: 1}'),
        ('model_nested_comp_with_scheduler.py', 'comp', '{A: 1}'),
        (
            'model_with_control.py',
            'comp',
            '{Input: [0.5, 0.123], reward: [20, 20]}'
        ),
        (
            'stroop_conflict_monitoring.py',
            'Stroop_model',
            str(stroop_stimuli).replace("'", '')
        )
    ]
)
def test_json_results_equivalence(
    filename,
    composition_name,
    input_dict_str,
):
    # Get python script from disk and execute
    full_filename = f'{os.path.dirname(__file__)}/{filename}'
    with open(full_filename, 'r') as orig_file:
        exec(orig_file.read())
        exec(f'{composition_name}.run(inputs={input_dict_str})')
        orig_results = eval(f'{composition_name}.results')

    # reset random seed
    pnl.core.globals.utilities.set_global_seed(0)

    # Generate python script from JSON summary of composition and execute
    json_summary = eval(f'{composition_name}.json_summary')
    exec(pnl.generate_script_from_json(json_summary))
    exec(f'{composition_name}.run(inputs={input_dict_str})')
    new_results = eval(f'{composition_name}.results')

    assert orig_results == new_results

    # Delete any traces from previous Compositions
    exec(f'del {composition_name}')
    # for mech in [m for m in locals().values() if isinstance(m, pnl.Mechanism)]:
    for mech in [m for m in locals().items() if isinstance(m[1], pnl.Mechanism)]:
        del locals()[mech[0]]
    exec(f'pnl.clear_registry(pnl.CompositionRegistry)')
    exec(f'pnl.clear_registry(pnl.MechanismRegistry)')

    json_filename = filename+'.json'
    with open(json_filename, 'w') as json_file:
        json_file.write(json_summary)
        pnl.read_json_file(filename=json_filename,
                           path=os.path.dirname(__file__)
                           )
        exec(f'{composition_name}.run(inputs={input_dict_str})')
        final_results = eval(f'{composition_name}.results')

    assert orig_results == final_results


