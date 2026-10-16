"""An installed compiler alone does not make the OpenMP backend available."""

from types import SimpleNamespace

import pytest

from psyneulink.core.batched.numerical import native


@pytest.fixture(autouse=True)
def clear_probe_cache():
    native._compiler_supports_openmp.cache_clear()
    yield
    native._compiler_supports_openmp.cache_clear()


@pytest.mark.parametrize("returncode, available", [(0, True), (1, False)])
def test_native_availability_checks_openmp_linking(monkeypatch, returncode, available):
    monkeypatch.setattr(native.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(native.sys, "platform", "darwin")
    monkeypatch.setattr(native.shutil, "which", lambda name: f"/bin/{name}")
    calls = []

    def compile_probe(command, **kwargs):
        calls.append(command)
        assert "-fopenmp" in command
        assert "omp_get_max_threads" in kwargs["input"]
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(native.subprocess, "run", compile_probe)
    assert native.native_kernels_available() is available
    assert native.native_kernels_available() is available
    assert len(calls) == 1


def test_windows_compiler_is_not_mistaken_for_gcc_backend(monkeypatch):
    monkeypatch.setattr(native.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(native.sys, "platform", "win32")
    monkeypatch.setattr(native.shutil, "which", lambda name: f"/bin/{name}")
    assert not native.native_kernels_available()


def test_native_availability_requires_torch(monkeypatch):
    monkeypatch.setattr(native.importlib.util, "find_spec", lambda name: None)
    assert not native.native_kernels_available()
