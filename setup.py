import versioneer

# Always prefer setuptools over distutils
from setuptools import find_packages, setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


def get_requirements(require_name=None):
    prefix = require_name + '_' if require_name is not None else ''
    with open(path.join(here, prefix + 'requirements.txt'), encoding='utf-8') as f:
        return f.read().strip().split('\n')


setup(
    name='psyneulink',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),

    description='A block modeling system for cognitive neuroscience',
    long_description=long_description,

    # Github address.
    url='https://github.com/PrincetonUniversity/PsyNeuLink',

    # Author details
    author='Jonathan Cohen, Princeton University, Intel',
    author_email='jdc@princeton.edu',

    # Choose your license
    license='Apache',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # This project is liscensed as follows
        'License :: OSI Approved :: Apache Software License',

        # Supported Python Versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    # Require recent python
    python_requires=">=3.6",

    # What does your project relate to?
    keywords='cognitive modeling',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=get_requirements(),

    extras_require={
        'dev': get_requirements('dev'),
        'doc': get_requirements('doc'),
        'tutorial': get_requirements('tutorial'),
    }
)
