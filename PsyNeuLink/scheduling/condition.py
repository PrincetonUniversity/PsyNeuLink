first_n_calls = lambda n: lambda calls: calls <= n
every_n_calls = lambda n: lambda calls: calls%n == 0