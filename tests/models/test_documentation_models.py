import json
import os
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
    'model_name, composition_name, additional_args, variant',
    [
        pytest.param(
            'Cohen_Huston1994',
            'Bidirectional_Stroop',
            [],
            None,
            marks=pytest.mark.stress
        ),
        pytest.param(
            'Cohen_Huston1994',
            'Bidirectional_Stroop',
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
            None,
            marks=pytest.mark.stress
        ),
        pytest.param(
            'Cohen_Huston1994_horse_race',
            'Bidirectional_Stroop',
            [
                '--word-runs=2',
                '--color-runs=1',
                '--threshold=0.5',
                '--settle-trials=10',
                '--pre-stimulus-trials=10'
            ],
            REDUCED
        ),
        pytest.param('GilzenratModel', 'task', ['--noise-stddev=0.0'], None),
        pytest.param('Kalanthroff_PCTC_2018', 'PCTC', [], None, marks=pytest.mark.stress),
        pytest.param('Kalanthroff_PCTC_2018', 'PCTC', ['--threshold=0.2', '--settle-trials=10'], REDUCED),
        pytest.param('MontagueDayanSejnowski96', 'comp_5a', ['--figure', '5a'], None),
        pytest.param('MontagueDayanSejnowski96', 'comp_5b', ['--figure', '5b'], None),
        pytest.param('MontagueDayanSejnowski96', 'comp_5c', ['--figure', '5c'], None),
        pytest.param('Nieuwenhuis2005Model', 'task', [], None),
    ]
)
def test_documentation_models(
    model_name,
    composition_name,
    additional_args,
    variant,
):
    models_dir = os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        'psyneulink',
        'library',
        'models'
    )
    model_file = os.path.join(models_dir, f'{model_name}.py')
    old_argv = sys.argv
    sys.argv = [model_file] + additional_args
    script_globals = runpy.run_path(model_file)
    sys.argv = old_argv

    suffix = f'-{variant}' if variant is not None else ''
    expected_results_file = os.path.join(
        models_dir,
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
