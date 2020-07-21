import psyneulink as pnl
import pytest

from psyneulink.core.scheduling.time import Time, TimeHistoryTree, TimeScale


class TestTime:
    @pytest.mark.parametrize(
        'base, increment_time_scale, expected',
        [
            (Time(run=0, trial=0, pass_=0, time_step=0), TimeScale.TRIAL, Time(run=0, trial=1, pass_=0, time_step=0)),
            (Time(run=0, trial=0, pass_=5, time_step=9), TimeScale.TRIAL, Time(run=0, trial=1, pass_=0, time_step=0)),
            (Time(run=1, trial=0, pass_=5, time_step=9), TimeScale.TRIAL, Time(run=1, trial=1, pass_=0, time_step=0)),
            (Time(run=1, trial=0, pass_=5, time_step=9), TimeScale.TIME_STEP, Time(run=1, trial=0, pass_=5, time_step=10)),
        ]
    )
    def test_increment(self, base, increment_time_scale, expected):
        base._increment_by_time_scale(increment_time_scale)
        assert base == expected

    def test_multiple_runs(self):
        t1 = pnl.TransferMechanism()
        t2 = pnl.TransferMechanism()

        C = pnl.Composition(pathways=[t1, t2])

        C.run(inputs={t1: [[1.0], [2.0], [3.0]]})
        assert C.scheduler.get_clock(C).time == Time(run=1, trial=0, pass_=0, time_step=0)

        C.run(inputs={t1: [[4.0], [5.0], [6.0]]})
        assert C.scheduler.get_clock(C).time == Time(run=2, trial=0, pass_=0, time_step=0)


class TestTimeHistoryTree:
    def test_defaults(self):
        h = TimeHistoryTree()

        for node in [h, h.children[0]]:
            assert len(node.children) == 1
            assert all([node.total_times[ts] == 0 for ts in node.total_times])
            assert node.time_scale == TimeScale.get_parent(node.children[0].time_scale)
            assert node.time_scale >= TimeScale.TRIAL

    @pytest.mark.parametrize(
        'max_depth',
        [
            (TimeScale.RUN),
            (TimeScale.TRIAL)
        ])
    def test_max_depth(self, max_depth):
        h = TimeHistoryTree(max_depth=max_depth)

        node = h
        found_max_depth = h.time_scale == max_depth
        while len(node.children) > 0:
            node = node.children[0]
            found_max_depth = found_max_depth or node.time_scale == max_depth
            assert node.time_scale >= max_depth

        assert found_max_depth
