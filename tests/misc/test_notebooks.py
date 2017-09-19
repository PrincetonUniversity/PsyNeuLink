"""Test that all notebooks in the repository run without error"""
import os
import subprocess


def _notebook_run(filepath):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    _, name = os.path.split(filepath)
    name = os.path.splitext(name)[0]

    # outfilename = os.path.join(outdir, '{}.ipynb'.format(name))
    args = ['jupyter', 'nbconvert', '--log-level', 'WARN',
            '--to', 'notebook', '--execute',
            '--ExecutePreprocessor.timeout=300', # 5 min max for executing
            '--stdout', filepath]
    return subprocess.check_call(args, stdout=subprocess.DEVNULL)

def _find_ipynbs():
    """Finds all the jupyter notebooks present in the repository"""
    root_dir = os.path.abspath(os.path.join(__file__, '../../..'))

    ipynb_filepaths = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.ipynb'):# and os.path.split(root)[1] != '.ipynb_checkpoints':
                ipynb_filepaths.append(os.path.join(root, filename))

    return ipynb_filepaths

def test_ipynb():
    for filepath in _find_ipynbs():
        print('Running {}'.format(filepath))
        _notebook_run(filepath)


if __name__ == '__main__':
    test_ipynb()
