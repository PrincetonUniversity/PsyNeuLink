import psyneulink as pnl
import numpy as np

class TestReduce:

    def test_single_array(self):
        R_function = pnl.Reduce(operation=pnl.SUM)
        R_mechanism = pnl.ProcessingMechanism(function=pnl.Reduce(operation=pnl.SUM),
                                              default_variable=[[1, 2, 3, 4, 5]],
                                              name="R_mechanism")

        assert np.allclose(R_function.execute([1, 2, 3, 4, 5]), [15.0])
        assert np.allclose(R_function.execute([[1, 2, 3, 4, 5]]), [15.0])
        assert np.allclose(R_function.execute([[[1, 2, 3, 4, 5]]]), [1, 2, 3, 4, 5])
        # assert np.allclose(R_function.execute([[[1, 2, 3, 4, 5]]]), [15.0])

        assert np.allclose(R_mechanism.execute([1, 2, 3, 4, 5]), [[15.0]])
        assert np.allclose(R_mechanism.execute([[1, 2, 3, 4, 5]]), [[15.0]])
        assert np.allclose(R_mechanism.execute([1, 2, 3, 4, 5]), [15.0])
        # assert np.allclose(R_mechanism.execute([[1, 2, 3, 4, 5]]), [15.0])

    def test_column_vector(self):
        R_function = pnl.Reduce(operation=pnl.SUM)
        R_mechanism = pnl.ProcessingMechanism(function=pnl.Reduce(operation=pnl.SUM),
                                              default_variable=[[1], [2], [3], [4], [5]],
                                              name="R_mechanism")

        assert np.allclose(R_function.execute([[1], [2], [3], [4], [5]]), [1, 2, 3, 4, 5])
        # assert np.allclose(R_function.execute([[1], [2], [3], [4], [5]]), [15.0])
        assert np.allclose(R_function.execute([[[1], [2], [3], [4], [5]]]), [15.0])

        assert np.allclose(R_mechanism.execute([[1], [2], [3], [4], [5]]), [1, 2, 3, 4, 5])
        # assert np.allclose(R_mechanism.execute([[1], [2], [3], [4], [5]]), [15.0])

    def test_matrix(self):
        R_function = pnl.Reduce(operation=pnl.SUM)
        R_mechanism = pnl.ProcessingMechanism(function=pnl.Reduce(operation=pnl.SUM),
                                              default_variable=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                              name="R_mechanism")

        assert np.allclose(R_function.execute([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [6, 15, 24])
        assert np.allclose(R_function.execute([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]), [12, 15, 18])

        assert np.allclose(R_mechanism.execute([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [6, 15, 24])

    # def test_heterogeneous_arrays(self):
    #     R_function = pnl.Reduce(operation=pnl.SUM)
    #     # R_mechanism = pnl.ProcessingMechanism(function=pnl.Reduce(operation=pnl.SUM),
    #     #                                       default_variable=[[1, 2], [3, 4, 5], [6, 7, 8, 9]],
    #     #                                       name="R_mechanism")
    #     print(R_function.execute([[1, 2], [3, 4, 5], [6, 7, 8, 9]]))
    #     print(R_function.execute([[[1, 2], [3, 4, 5], [6, 7, 8, 9]]]))
    #
    #     # print("mech = ", R_mechanism.execute([[1, 2], [3, 4, 5], [6, 7, 8, 9]]))
    #     # print("mech = ", R_mechanism.execute([[[1, 2], [3, 4, 5], [6, 7, 8, 9]]]))
    #     # print("mech = ", R_mechanism.execute([[[1, 2], [3, 4, 5], [6, 7, 8, 9]]]))
    #
