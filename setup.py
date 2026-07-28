# Project metadata and configuration live in pyproject.toml. This file exists
# only to wire up versioneer, which provides the dynamic version and the build
# cmdclass and requires a setup.py entry point.
import versioneer
from setuptools import setup

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
