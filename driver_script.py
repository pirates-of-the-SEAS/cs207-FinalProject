"""
Example driver script using Newton's Method to find the roots of sin

Note + iterable + autodiff yields collection of autodiffs.
have to make autodiffvector

"""

import pandas as pd
import numpy as np

from ARRRtomatic_diff import (AutoDiff, AutoDiffVector, AutoDiffRev,
                              AutoDiffRevVector)
from ARRRtomatic_diff.functions import sin, exp, sqrt, log
from ARRRtomatic_diff.optimization import (do_newtons_method,
                                           example_scalar,
                                           example_multivariate,
                                           do_bfgs,
                                           rosenbrock,
                                           parabola,
                                           do_gradient_descent,
                                           generate_nonlinear_lsq_data,
                                           beacon_resids,
                                           beacon_dist,
                                           do_levenberg_marquardt,
                                           example_loss,
                                           do_stochastic_gradient_descent
)



if __name__ == '__main__':
    # x = AutoDiffRev(name='x', val=5)
    # x2 = AutoDiffRev(name='x', val=4)
    # y = AutoDiffRev(name='y', val=4)

    # set([x])

    # x_f = AutoDiff(name='x', val=5)
    # y_f = AutoDiff(name='y', val=4)

    # # u_f = AutoDiffVector([
    # #     -y_f,
    # #     x_f
    # # ])

    # # v_f = AutoDiffVector([
    # #     x_f,
    # #     y_f
    # # ])

    # # u = AutoDiffRevVector([
    # #     -y,
    # #     x
    # # ])

    # # v = AutoDiffRevVector([
    # #     x,
    # #     y
    # # ])

    # a = x_f + y_f
    # b = x + y

    # print(a.get_gradient())
    # print(b.get_gradient())
    





    f1 = AutoDiff(name='x', val=-1)
    f2 = AutoDiff(name='y', val=3)
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector((-f2, f1))
    q = [2, 1.5]

    np.testing.assert_array_equal((u + q).val, [1, 4.5]), 'Addition failed'
    np.testing.assert_array_equal((q + u).val, [1, 4.5]), 'Addition failed'
    J, order = (u + q).get_jacobian()
    np.testing.assert_array_equal(J, [[1, 0], [0, 1]]), 'Addition failed'
    J, order = (v + q).get_jacobian()
    np.testing.assert_array_equal(J, [[0, -1], [1, 0]]), 'Addition failed'
    J, order = (q + u).get_jacobian()
    np.testing.assert_array_equal(J, [[1, 0], [0, 1]]), 'Addition failed'
    np.testing.assert_array_equal((u + v).val, [-4,  2]), 'Addition failed'
    np.testing.assert_array_equal((v + u).val, [-4, 2]), 'Addition failed'
    J, order = (u + v).get_jacobian()
    np.testing.assert_array_equal(J, [[1, -1], [1, 1]]), 'Addition failed'
