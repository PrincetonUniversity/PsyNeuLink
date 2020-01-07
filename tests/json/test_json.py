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
    filename = f'{os.path.dirname(__file__)}/{filename}'
    with open(filename, 'r') as orig_file:
        exec(orig_file.read())
        exec(f'{composition_name}.run(inputs={input_dict_str})')
        orig_results = eval(f'{composition_name}.results')

    # reset random seed
    pnl.core.globals.utilities.set_global_seed(0)

    exec(
        pnl.generate_script_from_json(
            eval(f'{composition_name}.json_summary')
        )
    )
    exec(f'{composition_name}.run(inputs={input_dict_str})')
    new_results = eval(f'{composition_name}.results')

    assert orig_results == new_results
