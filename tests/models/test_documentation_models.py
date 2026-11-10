import json
import pathlib
import runpy
import sys

import numpy as np
import pytest

import psyneulink as pnl

# Variants for test_documentation_models:

# parameters for the model were changed to reduce running time, and will
# not match the equivalent papers' results
REDUCED = 'reduced'


# NOTE: to add new models, run the script with valid results, then
# dump the results of a Composition to a file named
# {model_name}-{composition_name}.json in the same directory as
# the library models. Example for Cohen_Huston1994:
# json.dump(
#     Bidirectional_Stroop.results,
#     cls=pnl.PNLJSONEncoder,
#     fp=open('psyneulink/library/models/results/Cohen_Huston1994-Bidirectional_Stroop.json', 'w'),
#     indent=2,
# )
# Using prettier https://prettier.io/ can reduce the line footprint of
# the resulting file while not totally minifying it
@pytest.mark.parametrize(
    'model_name, composition_name, figure_args, additional_args, variant',
    [
        pytest.param(
            'Cohen_Huston1994',
            'Bidirectional_Stroop',
            [],
            [],
            None,
            marks=pytest.mark.stress
        ),
        pytest.param(
            'Cohen_Huston1994',
            'Bidirectional_Stroop',
            [],
            [
                '--threshold=0.5',
                '--settle-trials=10'
            ],
            REDUCED
        ),
        pytest.param(
            'Cohen_Huston1994_horse_race',
            'Bidirectional_Stroop',
            [],
            [],
            None,
            marks=pytest.mark.stress
        ),
        pytest.param(
            'Cohen_Huston1994_horse_race',
            'Bidirectional_Stroop',
            [],
            [
                '--word-runs=2',
                '--color-runs=1',
                '--threshold=0.5',
                '--settle-trials=10',
                '--pre-stimulus-trials=10'
            ],
            REDUCED
        ),
        pytest.param('GilzenratModel', 'task', [], ['--noise-stddev=0.0'], None),
        pytest.param('Kalanthroff_PCTC_2018', 'PCTC', [], [], None, marks=pytest.mark.stress),
        pytest.param('Kalanthroff_PCTC_2018', 'PCTC', [], ['--threshold=0.2', '--settle-trials=10'], REDUCED),
        pytest.param('MontagueDayanSejnowski96', 'comp_5a', ['5a'], [], None),
        pytest.param('MontagueDayanSejnowski96', 'comp_5b', ['5b'], [], None),
        pytest.param('MontagueDayanSejnowski96', 'comp_5c', ['5c'], [], None),
        pytest.param('Nieuwenhuis2005Model', 'task', [], [], None),
    ]
)
def test_documentation_models(
    model_name,
    composition_name,
    figure_args,
    additional_args,
    variant,
    tmp_path,
):
    models_dir = pathlib.Path(pnl.__file__).parent.joinpath('library', 'models')
    model_file = models_dir.joinpath(f'{model_name}.py')
    old_argv = sys.argv
    sys.argv = [model_file] + additional_args + ['--output-dir', str(tmp_path)]
    if figure_args:
        sys.argv += ['--figure'] + figure_args
    script_globals = runpy.run_path(model_file)
    sys.argv = old_argv

    suffix = f'-{variant}' if variant is not None else ''
    expected_results_file = models_dir.joinpath(
        'results',
        f'{model_name}-{composition_name}{suffix}.json'
    )
    with open(expected_results_file) as fi:
        expected_results = pnl.convert_all_elements_to_np_array(json.loads(fi.read()))

    results = pnl.convert_all_elements_to_np_array(script_globals[composition_name].results)

    assert expected_results.shape == results.shape
    np.testing.assert_allclose(
        pytest.helpers.expand_np_ndarray(expected_results),
        pytest.helpers.expand_np_ndarray(results)
    )

    fig_prefix = model_file.with_suffix('').name
    if not figure_args:
        figures = [f'{fig_prefix}.png']
    else:
        figures = [f'{fig_prefix}-{fig}.png' for fig in figure_args]

    for fig in figures:
        figure_path = tmp_path.joinpath(fig)
        assert figure_path.exists()
        # just test that there's some data, and more than a .png header
        # (plots vary by env or system, and results themselves are tested)
        assert figure_path.stat().st_size > 50
