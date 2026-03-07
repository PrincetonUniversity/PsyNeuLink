from typing import Any, Dict, Set, Union
import pytest

import psyneulink as pnl


def _get_expected_instances(names: Union[None, str, Set[str]], _locals: Dict[str, Any]):
    if names is not None:
        if isinstance(names, set):
            return {_locals[x] for x in names}
        else:
            return _locals[names]
    return None


class TestRegistry:
    # registry is not properly cleared in pytest --forked mode despite
    # clear_registry in pytest_runtest_teardown.
    # unclear if this is a bug or expected
    @pytest.fixture(autouse=True)
    def _preclear_registry(self):
        for registry in pnl.primary_registries:
            pnl.clear_registry(registry)

    @pytest.fixture(scope='function')
    def reg_test_components(self):
        apm = pnl.ProcessingMechanism(name='A')  # noqa: F841
        bpm = pnl.ProcessingMechanism(name='B')  # noqa: F841

        afunc = pnl.Logistic(name='A')  # noqa: F841

        return apm, bpm, afunc

    @pytest.mark.parametrize(
        'name_or_types, types, expected',
        [
            # name only, first arg
            ('A', None, {'apm', 'afunc'}),
            ('B', None, 'bpm'),
            ('C', None, None),
            # name and types
            ('A', pnl.ProcessingMechanism, 'apm'),
            ('A', pnl.Logistic, 'afunc'),
            ('A', pnl.Function, 'afunc'),
            (None, pnl.ProcessingMechanism, {'apm', 'bpm'}),
            ('A', pnl.Component, {'apm', 'afunc'}),
            ('C', pnl.Component, None),
            # type as first arg
            (pnl.ProcessingMechanism, None, {'apm', 'bpm'}),
            (pnl.ProcessingMechanism, pnl.Projection, {'apm', 'bpm'}),
            (pnl.Projection, None, None),
            (pnl.Projection, pnl.ProcessingMechanism, None),
            ((pnl.Mechanism, pnl.Logistic, pnl.Projection), None, {'apm', 'bpm', 'afunc'}),
        ],
    )
    def test_get_entry(self, reg_test_components, name_or_types, types, expected):
        apm, bpm, afunc = reg_test_components

        res = pnl.global_registry.get_entry(name_or_types, types)
        assert res == _get_expected_instances(expected, locals())

    @pytest.mark.parametrize(
        'types, exp_contains, exp_removed',
        [
            (pnl.Component, {}, {'A': {'apm', 'afunc'}, 'B': {'bpm'}}),
            ((pnl.Mechanism, pnl.Logistic, pnl.Composition), {}, {'A': {'apm', 'afunc'}, 'B': {'bpm'}}),
            (pnl.Projection, {'A': {'apm', 'afunc'}, 'B': {'bpm'}}, {}),
            (pnl.Function, {'A': {'apm'}, 'B': {'bpm'}}, {'A': {'afunc'}}),
        ],
    )
    def test_clear_entries(self, reg_test_components, types, exp_contains, exp_removed):
        apm, bpm, afunc = reg_test_components

        pnl.global_registry.clear_entries(types)
        test_locals = locals()
        exp_contains = {name: _get_expected_instances(exp_contains[name], test_locals) for name in exp_contains}
        exp_removed = {name: _get_expected_instances(exp_removed[name], test_locals) for name in exp_removed}

        for name, objs in exp_contains.items():
            assert (
                name in pnl.global_registry._instances
                and set(pnl.global_registry._instances[name]).intersection(objs) == objs
            )

        for name, objs in exp_removed.items():
            assert (
                name not in pnl.global_registry._instances
                or len(set(pnl.global_registry._instances[name]).intersection(objs)) == 0
            )

    def test_clear_entries_all(self, reg_test_components):
        apm, bpm, afunc = reg_test_components  # noqa: F841, RUF059

        pnl.global_registry.clear_entries()
        assert pnl.global_registry._instances == {}

    @pytest.mark.parametrize(
        'new_name, old_name_or_entry, types, exp_renamed',
        [
            ('B', 'A', None, {'apm', 'afunc'}),
            ('B', 'A', (pnl.Projection, pnl.Function), {'afunc'}),
            ('B-1', 'B', pnl.Function, set()),  # types not ignored since old_name_or_entry is a name
            ('B', 'A-1', None, set()),
            ('B', 'A-1', pnl.Component, set()),
            ('B', 'apm', None, {'apm'}),
            ('B', 'apm', pnl.Function, {'apm'}),  # types ignored since old_name_or_entry is an object
            ('B', 'afunc', pnl.Function, {'afunc'}),
        ],
    )
    def test_rename_instance_in_registry(
        self, reg_test_components, new_name, old_name_or_entry, types, exp_renamed
    ):
        apm, bpm, afunc = reg_test_components
        try:
            old_name_or_entry = _get_expected_instances(old_name_or_entry, locals())
        except KeyError:
            pass

        pnl.global_registry.rename_instance_in_registry(new_name, old_name_or_entry, types)

        exp_renamed = _get_expected_instances(exp_renamed, locals())
        not_renamed = set(reg_test_components).difference(exp_renamed)

        for obj in exp_renamed:
            old_name = obj.name
            assert new_name in pnl.global_registry._instances
            assert obj in pnl.global_registry._instances[new_name]
            assert (
                old_name not in pnl.global_registry._instances
                or len(pnl.global_registry._instances[old_name]) > 0
            )

        for obj in not_renamed:
            assert obj.name in pnl.global_registry._instances
            assert obj in pnl.global_registry._instances[obj.name]

    @pytest.mark.parametrize(
        'name_or_entry, types, exp_removed',
        [
            ('A', pnl.Component, {'apm', 'afunc'}),
            ('A', pnl.Projection, set()),  # types not ignored since name_or_entry is a name
            ('A', None, {'apm', 'afunc'}),
            ('A', (pnl.Projection, pnl.Mechanism), {'apm'}),
            ('apm', pnl.Component, {'apm'}),
            ('apm', pnl.Function, {'apm'}),  # types ignored since name_or_entry is an object
            ('nonexistent name', None, set()),
            ('nonexistent name', pnl.Component, set()),
        ],
    )
    def test_remove_instance_from_registry(
        self, reg_test_components, name_or_entry, types, exp_removed
    ):
        apm, bpm, afunc = reg_test_components
        try:
            name_or_entry = _get_expected_instances(name_or_entry, locals())
        except KeyError:
            pass

        pnl.global_registry.remove_instance_from_registry(name_or_entry, types)

        exp_removed = _get_expected_instances(exp_removed, locals())
        not_removed = set(reg_test_components).difference(exp_removed)

        for obj in exp_removed:
            if obj.name in pnl.global_registry._instances:
                assert obj not in pnl.global_registry._instances[obj.name]
                assert len(pnl.global_registry._instances[obj.name]) > 0

        for obj in not_removed:
            assert obj.name in pnl.global_registry._instances
            assert obj in pnl.global_registry._instances[obj.name]
