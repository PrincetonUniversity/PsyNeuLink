import numpy as np
import os
import psyneulink as pnl
import pytest
import sys


pytest.importorskip(
    'modeci_mdf',
    reason='JSON methods require modeci_mdf package'
)

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
    ('model_basic.py', 'comp', '{A: 1}', True),
    ('model_basic.py', 'comp', '{A: 1}', False),
    ('model_basic_non_identity.py', 'comp', '{A: 1}', True),
    ('model_basic_non_identity.py', 'comp', '{A: 1}', False),
    ('model_udfs.py', 'comp', '{A: 10}', True),
    ('model_udfs.py', 'comp', '{A: 10}', False),
    ('model_varied_matrix_sizes.py', 'comp', '{A: [1, 2]}', True),
    ('model_varied_matrix_sizes.py', 'comp', '{A: [1, 2]}', False),
    ('model_integrators.py', 'comp', '{A: 1.0}', True),
    ('model_integrators.py', 'comp', '{A: 1.0}', False),
    pytest.param(
        'model_nested_comp_with_scheduler.py',
        'comp',
        '{A: 1}',
        True,
        marks=pytest.mark.xfail(reason='Nested Graphs not supported in MDF')
    ),
    pytest.param(
        'model_nested_comp_with_scheduler.py',
        'comp',
        '{A: 1}',
        False,
        marks=pytest.mark.xfail(reason='Nested Graphs not supported in MDF')
    ),
    (
        'model_with_control.py',
        'comp',
        '{Input: [0.5, 0.123], reward: [20, 20]}',
        False
    ),
    (
        'stroop_conflict_monitoring.py',
        'Stroop_model',
        str(stroop_stimuli).replace("'", ''),
        False
    ),
    ('model_backprop.py', 'comp', '{a: [1, 2, 3]}', False),
]


@pytest.mark.parametrize(
    'filename, composition_name, input_dict_str, simple_edge_format',
    json_results_parametrization
)
def test_json_results_equivalence(
    filename,
    composition_name,
    input_dict_str,
    simple_edge_format,
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
    json_summary = pnl.generate_json(eval(f'{composition_name}'), simple_edge_format=simple_edge_format)
    exec(pnl.generate_script_from_json(json_summary))
    exec(f'{composition_name}.run(inputs={input_dict_str})')
    new_results = eval(f'{composition_name}.results')
    assert pnl.safe_equals(orig_results, new_results)


@pytest.mark.parametrize(
    'filename, composition_name, input_dict_str, simple_edge_format',
    json_results_parametrization
)
def test_write_json_file(
    filename,
    composition_name,
    input_dict_str,
    simple_edge_format,
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
    exec(f'pnl.write_json_file({composition_name}, json_filename, simple_edge_format=simple_edge_format)')
    exec(pnl.generate_script_from_json(json_filename))
    # exec(f'{composition_name}.run(inputs={input_dict_str})')
    exec(f'pnl.get_compositions()[0].run(inputs={input_dict_str})')
    final_results = eval(f'{composition_name}.results')
    assert pnl.safe_equals(orig_results, final_results)


@pytest.mark.parametrize(
    'filename, input_dict_strs',
    [
        pytest.param(
            'model_with_two_conjoint_comps.py',
            {'comp': '{A: 1}', 'comp2': '{A: 1}'},
            marks=pytest.mark.xfail
        ),
        ('model_with_two_disjoint_comps.py', {'comp': '{A: 1}', 'comp2': '{C: 1}'}),
    ]
)
def test_write_json_file_multiple_comps(
    filename,
    input_dict_strs,
):
    orig_results = {}

    # Get python script from file and execute
    filename = f'{os.path.dirname(__file__)}/{filename}'
    with open(filename, 'r') as orig_file:
        exec(orig_file.read())

        for composition_name in input_dict_strs:
            exec(f'{composition_name}.run(inputs={input_dict_strs[composition_name]})')
            orig_results[composition_name] = eval(f'{composition_name}.results')

    # reset random seed
    pnl.core.globals.utilities.set_global_seed(0)

    # Save json_summary of Composition to file and read back in.
    json_filename = filename.replace('.py', '.json')

    exec(f'pnl.write_json_file([{",".join(input_dict_strs)}], json_filename)')
    exec(pnl.generate_script_from_json(json_filename))

    for composition_name in input_dict_strs:
        exec(f'{composition_name}.run(inputs={input_dict_strs[composition_name]})')
        final_results = eval(f'{composition_name}.results')
        assert orig_results[composition_name] == final_results, f'{composition_name}:'


# These runtime_params are necessary because noise seeding is not
# replicable between numpy and onnx.
# Values are generated from running onnx function RandomUniform and
# RandomNormal with parameters used in model_integrators.py (seed 0).
# RandomNormal values are different on mac versus linux and windows
if sys.platform == 'linux':
    onnx_integrators_fixed_seeded_noise = {
        'A': [[-0.9999843239784241]],
        'B': [[-1.1295466423034668]],
        'C': [[-0.0647732987999916]],
        'D': [[-0.499992161989212]],
        'E': [[-0.2499941289424896]],
    }
elif sys.platform == 'win32':
    onnx_integrators_fixed_seeded_noise = {
        'A': [[0.0976270437240601]],
        'B': [[-0.4184607267379761]],
        'C': [[0.290769636631012]],
        'D': [[0.04881352186203]],
        'E': [[0.1616101264953613]],
    }
else:
    assert sys.platform == 'darwin'
    onnx_integrators_fixed_seeded_noise = {
        'A': [[-0.9999550580978394]],
        'B': [[-0.8846577405929565]],
        'C': [[0.0576711297035217]],
        'D': [[-0.4999775290489197]],
        'E': [[-0.2499831467866898]],
    }

integrators_runtime_params = (
    'runtime_params={'
    + ','.join([f'{k}: {{ "noise": {v} }}' for k, v in onnx_integrators_fixed_seeded_noise.items()])
    + '}'
)


@pytest.mark.parametrize(
    'filename, composition_name, input_dict, simple_edge_format, run_args',
    [
        ('model_basic.py', 'comp', {'A': [[1.0]]}, True, ''),
        ('model_basic.py', 'comp', {'A': 1}, False, ''),
        ('model_basic_non_identity.py', 'comp', {'A': [[1.0]]}, True, ''),  # requires simple edges
        ('model_udfs.py', 'comp', {'A': [[10.0]]}, True, ''),
        ('model_udfs.py', 'comp', {'A': 10}, False, ''),
        ('model_varied_matrix_sizes.py', 'comp', {'A': [[1.0, 2.0]]}, True, ''),
        ('model_integrators.py', 'comp', {'A': 1.0}, True, integrators_runtime_params),
        ('model_integrators.py', 'comp', {'A': 1.0}, False, integrators_runtime_params),
    ]
)
def test_mdf_equivalence(filename, composition_name, input_dict, simple_edge_format, run_args):
    from modeci_mdf.utils import load_mdf
    import modeci_mdf.execution_engine as ee

    # Get python script from file and execute
    filename = f'{os.path.dirname(__file__)}/{filename}'
    with open(filename, 'r') as orig_file:
        exec(orig_file.read())
        inputs_str = str(input_dict).replace("'", '')
        exec(f'{composition_name}.run(inputs={inputs_str}, {run_args})')
        orig_results = eval(f'{composition_name}.results')

    # Save json_summary of Composition to file and read back in.
    json_filename = filename.replace('.py', '.json')
    pnl.write_json_file(eval(composition_name), json_filename, simple_edge_format=simple_edge_format)

    m = load_mdf(json_filename)
    eg = ee.EvaluableGraph(m.graphs[0], verbose=True)
    eg.evaluate(initializer={f'{node}_InputPort_0': i for node, i in input_dict.items()})

    mdf_results = [
        [eo.curr_value for _, eo in eg.enodes[node.id].evaluable_outputs.items()]
        for node in eg.scheduler.consideration_queue[-1]
    ]

    assert pnl.safe_equals(orig_results, mdf_results)
