import psyneulink as pnl


def test_Stability_squeezes_variable():
    s1 = pnl.Stability(default_variable=[[0.0, 0.0]])
    s2 = pnl.Stability(default_variable=[0.0, 0.0])

    assert s1.execute([5, 5]) == s2.execute([5, 5])
