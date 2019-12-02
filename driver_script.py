"""
Example driver script using Newton's Method to find the roots of sin
"""

import numpy as np

from ARRRtomatic_diff import AutoDiff, AutoDiffVector
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
                                           do_levenberg_marquardt
)



if __name__ == '__main__':
    X, y = generate_nonlinear_lsq_data()


    b = do_levenberg_marquardt(np.array([0.4,0.9]),
                           beacon_resids,
                           X,
                           y,
                           mu=None,
                           S=None,
                           tol=1e-8,
                           max_iter=2000,
                           verbose=0)

   

    print(b)


