
K = {'A': 0}

def run_games(cost_rate):
    K['A'] = K['A'] + 1
    return K['A']

if __name__ == "__main__":
    print(run_games(-.8))
