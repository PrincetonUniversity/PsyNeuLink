import copy
import numpy as np
import os
import psyneulink as pnl
import pytest


pytest.importorskip(
    'modeci_mdf',
    reason='MDF methods require modeci_mdf package'
)
from modeci_mdf.execution_engine import evaluate_onnx_expr  # noqa: E402


def get_onnx_fixed_noise_str(onnx_op, **kwargs):
    # high precision printing needed because script will be executed from string
    # 16 is insufficient on windows
    with np.printoptions(precision=32):
        return str(
            evaluate_onnx_expr(f'onnx_ops.{onnx_op}', base_parameters=kwargs, evaluated_parameters=kwargs)
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
    filename = os.path.join(os.path.dirname(__file__), filename)
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
    filename = os.path.join(os.path.dirname(__file__), filename)
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
    filename = os.path.join(os.path.dirname(__file__), filename)
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


def _get_mdf_model_results(evaluable_graph):
    return [
        [eo.curr_value for _, eo in evaluable_graph.enodes[node.id].evaluable_outputs.items()]
        for node in evaluable_graph.scheduler.consideration_queue[-1]
    ]


# These runtime_params are necessary because noise seeding is not
# replicable between numpy and onnx.
# Values are generated from running onnx function RandomUniform and
# RandomNormal with parameters used in model_integrators.py (seed 0).
# RandomNormal values are different on mac versus linux and windows
onnx_noise_data = {
    'randomuniform': {
        'A': {'low': -1.0, 'high': 1.0, 'seed': 0, 'shape': (1, 1)},
        'D': {'low': -0.5, 'high': 0.5, 'seed': 0, 'shape': (1, 1)},
        'E': {'low': -0.25, 'high': 0.5, 'seed': 0, 'shape': (1, 1)}
    },
    'randomnormal': {
        'B': {'mean': -1.0, 'scale': 0.5, 'seed': 0, 'shape': (1, 1)},
        'C': {'mean': 0.0, 'scale': 0.25, 'seed': 0, 'shape': (1, 1)},
    }
}
onnx_integrators_fixed_seeded_noise = {}
integrators_runtime_params = None

for func_type in onnx_noise_data:
    for node, args in onnx_noise_data[func_type].items():
        # generates output from onnx noise functions with seed 0 to be
        # passed in in runtime_params during psyneulink execution
        onnx_integrators_fixed_seeded_noise[node] = get_onnx_fixed_noise_str(func_type, **args)

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
    filename = os.path.join(os.path.dirname(__file__), filename)
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

    assert pnl.safe_equals(orig_results, _get_mdf_model_results(eg))


ddi_termination_conds = [
    None,
    (
        "pnl.Or("
        "pnl.Threshold(A, parameter='value', threshold=A.function.defaults.threshold, comparator='>=', indices=(0,)),"
        "pnl.Threshold(A, parameter='value', threshold=-1 * A.function.defaults.threshold, comparator='<=', indices=(0,))"
        ")"
    ),
    'pnl.AfterNCalls(A, 10)',
]

# construct test data manually instead of with multiple @pytest.mark.parametrize
# so that other functions can use more appropriate termination conds
individual_functions_ddi_test_data = [
    (
        pnl.IntegratorMechanism,
        pnl.DriftDiffusionIntegrator(rate=0.5, offset=1, non_decision_time=1, seed=0),
        "{{A: {{'random_draw': {0} }} }}".format(get_onnx_fixed_noise_str('randomnormal', mean=0, scale=1, seed=0, shape=(1,)))
    ) + (x,)
    for x in ddi_termination_conds
]
individual_functions_fhn_test_data = [
    (
        pnl.IntegratorMechanism,
        pnl.FitzHughNagumoIntegrator,
        'None',
        'pnl.AfterNCalls(A, 10)',
    ),
    (
        pnl.IntegratorMechanism,
        pnl.FitzHughNagumoIntegrator(integration_method='EULER'),
        'None',
        'pnl.AfterNCalls(A, 10)',
    )
]


@pytest.mark.parametrize(
    'mech_type, function, runtime_params, trial_termination_cond',
    [
        *individual_functions_ddi_test_data,
        *individual_functions_fhn_test_data,
    ]
)
def test_mdf_equivalence_individual_functions(mech_type, function, runtime_params, trial_termination_cond):
    import modeci_mdf.execution_engine as ee

    A = mech_type(name='A', function=copy.deepcopy(function))
    comp = pnl.Composition(pathways=[A])

    try:
        trial_termination_cond = eval(trial_termination_cond)
    except TypeError:
        pass
    comp.scheduler.termination_conds = {pnl.TimeScale.TRIAL: trial_termination_cond}

    comp.run(inputs={A: [[1.0]]}, runtime_params=eval(runtime_params))

    model = pnl.get_mdf_model(comp)

    eg = ee.EvaluableGraph(model.graphs[0], verbose=True)
    eg.evaluate(initializer={'A_InputPort_0': 1.0})

    assert pnl.safe_equals(comp.results, _get_mdf_model_results(eg))


@pytest.mark.parametrize('filename', ['model_basic.py'])
@pytest.mark.parametrize('fmt', ['json', 'yml'])
def test_generate_script_from_mdf(filename, fmt):
    filename = os.path.join(os.path.dirname(__file__), filename)
    outfi = filename.replace('.py', f'.{fmt}')

    with open(filename, 'r') as orig_file:
        exec(orig_file.read())
        serialized = eval(f'pnl.get_mdf_serialized(comp, fmt="{fmt}")')

    with open(outfi, 'w') as f:
        f.write(serialized)

    with open(outfi, 'r') as f:
        assert pnl.generate_script_from_mdf(f.read()) == pnl.generate_script_from_mdf(outfi)
