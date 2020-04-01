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


json_results_parametrization = [
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


@pytest.mark.parametrize(
    'filename, composition_name, input_dict_str',
    json_results_parametrization
)
def test_json_results_equivalence(
    filename,
    composition_name,
    input_dict_str,
):
    # Get python script from file and execute
    filename = f'{os.path.dirname(__file__)}/{filename}'
    with open(filename, 'r') as orig_file:
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


@pytest.mark.parametrize(
    'filename, composition_name, input_dict_str',
    json_results_parametrization
)
def test_write_json_file(
    filename,
    composition_name,
    input_dict_str,
):
    # Get python script from file and execute
    filename = f'{os.path.dirname(__file__)}/{filename}'
    with open(filename, 'r') as orig_file:
        exec(orig_file.read())
        exec(f'{composition_name}.run(inputs={input_dict_str})')
        orig_results = eval(f'{composition_name}.results')

    # reset random seed
    pnl.core.globals.utilities.set_global_seed(0)

    # Save json_summary of Composition to file and read back in.
    json_filename = filename.replace('.py','.json')
    exec(f'pnl.write_json_file({composition_name}, json_filename)')
    exec(pnl.generate_script_from_json(json_filename))
    # exec(f'{composition_name}.run(inputs={input_dict_str})')
    exec(f'pnl.get_compositions()[0].run(inputs={input_dict_str})')
    final_results = eval(f'{composition_name}.results')
    assert orig_results == final_results
