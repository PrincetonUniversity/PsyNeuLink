def pytest_assertrepr_compare(op, left, right):
    if isinstance(left, list) and isinstance(right, list) and op == '==':
        return [
            'Time Step output matching:',
            'Actual output:', str(left),
            'Expected output:', str(right)
        ]
