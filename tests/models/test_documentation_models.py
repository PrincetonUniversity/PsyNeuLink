import json
import os
import runpy
import sys

import numpy as np
import pytest

import psyneulink as pnl


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
    'model_name, composition_name, additional_args, reduced',
    [
        pytest.param(
            'Cohen_Huston1994',
            'Bidirectional_Stroop',
            [],
            False,
            marks=pytest.mark.stress
        ),
        pytest.param(
            'Cohen_Huston1994',
            'Bidirectional_Stroop',
            [
                '--threshold=0.5',
                '--settle-trials=10'
            ],
            True
        ),
        pytest.param(
            'Cohen_Huston1994_horse_race',
            'Bidirectional_Stroop',
            [],
            False,
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
            True
        ),
        pytest.param('GilzenratModel', 'task', ['--noise-stddev=0.0'], False),
        pytest.param('Kalanthroff_PCTC_2018', 'PCTC', [], False, marks=pytest.mark.stress),
        pytest.param('Kalanthroff_PCTC_2018', 'PCTC', ['--threshold=0.2', '--settle-trials=10'], True),
        pytest.param('MontagueDayanSejnowski96', 'comp_5a', ['--figure', '5a'], False),
        pytest.param('MontagueDayanSejnowski96', 'comp_5b', ['--figure', '5b'], False),
        pytest.param('MontagueDayanSejnowski96', 'comp_5c', ['--figure', '5c'], False),
        pytest.param('Nieuwenhuis2005Model', 'task', [], False),
    ]
)
def test_documentation_models(
    model_name,
    composition_name,
    additional_args,
    reduced,
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
    sys.argv = [model_file, '--no-plot'] + additional_args
    script_globals = runpy.run_path(model_file)
    sys.argv = old_argv

    expected_results_file = os.path.join(
        models_dir,
        'results',
        f'{model_name}-{composition_name}{"-reduced" if reduced else ""}.json'
    )
    with open(expected_results_file) as fi:
        expected_results = pnl.convert_all_elements_to_np_array(json.loads(fi.read()))

    results = pnl.convert_all_elements_to_np_array(script_globals[composition_name].results)

    assert expected_results.shape == results.shape
    np.testing.assert_allclose(
        pytest.helpers.expand_np_ndarray(expected_results),
        pytest.helpers.expand_np_ndarray(results)
    )
