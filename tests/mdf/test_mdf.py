import copy
import numpy as np
import os
import sys
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


pnl_mdf_results_parametrization = [
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
    ('model_backprop.py', 'comp', '{A: [1, 2, 3]}', False),
]


def get_mdf_output_file(orig_filename, tmp_path, format='json'):
    """
    Returns:
        tuple(pathlib.Path, str, str):
            - a pytest tmp_path temp file using **orig_filename** and
              **format**
            - the full path to the temp file
            - the full path to the temp file formatted so that it can be
              used in an exec/eval string
    """
    mdf_file = tmp_path / orig_filename.replace('.py', f'.{format}')
    mdf_fname = str(mdf_file.absolute())

    # need to escape backslash to use a filename in exec on windows
    if sys.platform.startswith('win'):
        mdf_exec_fname = mdf_fname.replace('\\', '\\\\')
    else:
        mdf_exec_fname = mdf_fname

    return mdf_file, mdf_fname, mdf_exec_fname


def read_defined_model_script(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)

    with open(filename, 'r') as orig_file:
        model_input = orig_file.read()

    return model_input


def get_loaded_model_state(model_input: str):
    _globals = copy.copy(globals())
    _locals = copy.copy(locals())

    exec(model_input, _globals, _locals)

    return _globals, _locals


def run_compositions_in_state(
    composition_input_strs, _globals, _locals, extra_run_args_str=''
):
    results = {}

    for comp_name, inputs in composition_input_strs.items():
        exec(f'{comp_name}.run(inputs={inputs}, {extra_run_args_str})', _globals, _locals)
        results[comp_name] = eval(f'{comp_name}.results', _globals, _locals)

    return results, _globals, _locals


def get_model_results_and_state(
    model_input: str, composition_input_strs, extra_run_args_str=''
):
    _globals, _locals = get_loaded_model_state(model_input)
    return run_compositions_in_state(
        composition_input_strs, _globals, _locals, extra_run_args_str
    )


def assert_result_equality(orig_results, new_results):
    # compositions
    assert orig_results.keys() == new_results.keys()

    for comp_name in orig_results:
        np.testing.assert_allclose(
            orig_results[comp_name],
            new_results[comp_name],
            err_msg=f"Results for composition '{comp_name}' are not equal:"
        )


@pytest.mark.parametrize(
    'filename, composition_name, input_dict_str, simple_edge_format',
    pnl_mdf_results_parametrization
)
def test_get_mdf_serialized_results_equivalence_pnl_only(
    filename,
    composition_name,
    input_dict_str,
    simple_edge_format,
):
    comp_inputs = {composition_name: input_dict_str}

    # Get python script from file and execute
    orig_script = read_defined_model_script(filename)
    orig_results, orig_globals, orig_locals = get_model_results_and_state(
        orig_script, comp_inputs
    )

    # reset random seed
    pnl.core.globals.utilities.set_global_seed(0)
    # Generate python script from MDF serialization of composition and execute
    mdf_data = pnl.get_mdf_serialized(
        eval(f'{composition_name}', orig_globals, orig_locals),
        simple_edge_format=simple_edge_format
    )
    new_script = pnl.generate_script_from_mdf(mdf_data)
    new_results, _, _ = get_model_results_and_state(new_script, comp_inputs)

    assert_result_equality(orig_results, new_results)


@pytest.mark.parametrize(
    'filename, composition_name, input_dict_str, simple_edge_format',
    pnl_mdf_results_parametrization
)
def test_write_mdf_file_results_equivalence_pnl_only(
    filename,
    composition_name,
    input_dict_str,
    simple_edge_format,
    tmp_path,
):
    comp_inputs = {composition_name: input_dict_str}

    # Get python script from file and execute
    orig_script = read_defined_model_script(filename)
    orig_results, orig_globals, orig_locals = get_model_results_and_state(
        orig_script, comp_inputs
    )

    # reset random seed
    pnl.core.globals.utilities.set_global_seed(0)

    # Save MDF serialization of Composition to file and read back in.
    _, mdf_fname, mdf_exec_fname = get_mdf_output_file(filename, tmp_path)
    exec(
        f'pnl.write_mdf_file({composition_name}, "{mdf_exec_fname}", simple_edge_format={simple_edge_format})',
        orig_globals,
        orig_locals,
    )

    new_script = pnl.generate_script_from_mdf(mdf_fname)
    new_results, _, _ = get_model_results_and_state(new_script, comp_inputs)

    assert_result_equality(orig_results, new_results)


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
def test_write_mdf_file_results_equivalence_pnl_only_multiple_comps(
    filename,
    input_dict_strs,
    tmp_path,
):
    # Get python script from file and execute
    orig_script = read_defined_model_script(filename)
    orig_results, orig_globals, orig_locals = get_model_results_and_state(
        orig_script, input_dict_strs
    )
    # reset random seed
    pnl.core.globals.utilities.set_global_seed(0)

    # Save MDF serialization of Composition to file and read back in.
    _, mdf_fname, mdf_exec_fname = get_mdf_output_file(filename, tmp_path)
    exec(
        f'pnl.write_mdf_file([{",".join(input_dict_strs)}], "{mdf_exec_fname}")',
        orig_globals,
        orig_locals
    )

    new_script = pnl.generate_script_from_mdf(mdf_fname)
    new_results, _, _ = get_model_results_and_state(new_script, input_dict_strs)

    assert_result_equality(orig_results, new_results)


def _get_mdf_model_results(evaluable_graph, composition=None):
    """
    Returns psyneulink-style output for **evaluable_graph**, optionally
    casting outputs to their equivalent node's shape in **composition**
    """
    if composition is not None:
        node_output_shapes = {
            # NOTE: would use defaults.value here, but it doesn't always
            # match the shape of value (specifically here,
            # FitzHughNagumoIntegrator EULER)
            pnl.parse_valid_identifier(node.name): node.value.shape
            for node in composition.get_nodes_by_role(pnl.NodeRole.OUTPUT)
        }
    else:
        node_output_shapes = {}

    res = []
    for node in evaluable_graph.scheduler.consideration_queue[-1]:
        next_res_elem = [
            eo.curr_value for eo in evaluable_graph.enodes[node.id].evaluable_outputs.values()
        ]
        try:
            next_res_elem = np.reshape(next_res_elem, node_output_shapes[node.id])
        except KeyError:
            pass

        res.append(next_res_elem)

    return pnl.convert_to_np_array(res)


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
def test_mdf_pnl_results_equivalence(filename, composition_name, input_dict, simple_edge_format, run_args, tmp_path):
    from modeci_mdf.utils import load_mdf
    import modeci_mdf.execution_engine as ee

    comp_inputs = {composition_name: input_dict}

    # Get python script from file and execute
    orig_script = read_defined_model_script(filename)
    orig_results, orig_globals, orig_locals = get_model_results_and_state(
        orig_script, comp_inputs, run_args
    )

    # Save MDF serialization of Composition to file and read back in.
    _, mdf_fname, _ = get_mdf_output_file(filename, tmp_path)
    composition = eval(composition_name, orig_globals, orig_locals)
    pnl.write_mdf_file(composition, mdf_fname, simple_edge_format=simple_edge_format)

    m = load_mdf(mdf_fname)
    eg = ee.EvaluableGraph(m.graphs[0], verbose=True)
    eg.evaluate(initializer={f'{node}_InputPort_0': i for node, i in input_dict.items()})

    assert_result_equality(orig_results, {composition_name: _get_mdf_model_results(eg, composition)})


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
def test_mdf_pnl_results_equivalence_individual_functions(mech_type, function, runtime_params, trial_termination_cond):
    import modeci_mdf.execution_engine as ee

    A = mech_type(name='A', function=copy.deepcopy(function))
    comp = pnl.Composition(pathways=[A])

    try:
        trial_termination_cond = eval(trial_termination_cond)
    except TypeError:
        pass
    if trial_termination_cond is not None:
        comp.scheduler.termination_conds = {pnl.TimeScale.TRIAL: trial_termination_cond}

    comp.run(inputs={A: [[1.0]]}, runtime_params=eval(runtime_params))

    model = pnl.get_mdf_model(comp)

    eg = ee.EvaluableGraph(model.graphs[0], verbose=True)
    eg.evaluate(initializer={'A_InputPort_0': 1.0})

    np.testing.assert_array_equal(comp.results, _get_mdf_model_results(eg, comp))


@pytest.mark.parametrize(
    'filename, composition_name',
    [
        ('model_basic.py', 'comp'),
    ]
)
@pytest.mark.parametrize('fmt', ['json', 'yml'])
def test_generate_script_from_mdf(filename, composition_name, fmt, tmp_path):
    orig_file = read_defined_model_script(filename)
    exec(orig_file)
    serialized = eval(f'pnl.get_mdf_serialized({composition_name}, fmt="{fmt}")')

    mdf_file, mdf_fname, _ = get_mdf_output_file(filename, tmp_path, fmt)
    mdf_file.write_text(serialized)

    assert pnl.generate_script_from_mdf(mdf_file.read_text()) == pnl.generate_script_from_mdf(mdf_fname)
