def pytest_runtest_setup():
    import doctest
    doctest.ELLIPSIS_MARKER = "[...]"