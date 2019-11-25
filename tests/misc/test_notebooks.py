"""Test that all notebooks in the tutorial run without error"""
import os
import pytest
import subprocess
import sys


def _notebook_run(filepath):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    # _, name = os.path.split(filepath)
    # name = os.path.splitext(name)[0]

    # outfilename = os.path.join(outdir, '{}.ipynb'.format(name))
    args = [sys.executable, '-m', 'jupyter', 'nbconvert', '--log-level', 'WARN',
            '--to', 'notebook', '--execute',
            '--ExecutePreprocessor.timeout=300', # 5 min max for executing
            '--stdout', filepath]
    return subprocess.check_call(args, stdout=subprocess.DEVNULL)

def _find_ipynbs():
    """Finds all the jupyter notebooks present in the tutorial directory"""
    tutorial_dir = os.path.abspath(os.path.join(__file__, '../../../tutorial'))

    ipynb_filepaths = []
    for root, dirs, files in os.walk(tutorial_dir):
        for filename in files:
            if filename.endswith('.ipynb') and os.path.split(root)[1] != '.ipynb_checkpoints':
                ipynb_filepaths.append(os.path.join(root, filename))

    return ipynb_filepaths


@pytest.mark.parametrize("filepath", _find_ipynbs(), ids=os.path.basename)
def test_ipynb(filepath):
    old_python_path = os.getenv("PYTHONPATH")
    # pytest populates sys.path to access the tested module
    os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)
    _notebook_run(filepath)
    if old_python_path is not None:
        os.environ["PYTHONPATH"] = old_python_path
    else:
        del os.environ["PYTHONPATH"]

if __name__ == '__main__':
    for filepath in _find_ipynbs():
        print('Running {}'.format(filepath))
        _notebook_run(filepath)
