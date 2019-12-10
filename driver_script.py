"""
Example driver script using Newton's Method to find the roots of sin

Note + iterable + autodiff yields collection of autodiffs.
have to make autodiffvector

"""

import pandas as pd
import numpy as np

from ARRRtomatic_diff import (AutoDiff, AutoDiffVector, AutoDiffRev,
                              AutoDiffRevVector)
from ARRRtomatic_diff.functions import sin, exp, sqrt, log, tan
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
    f1 = AutoDiffRev(name='x', val=-2)
    f2 = AutoDiffRev(name='y', val=np.pi/8)
    u = AutoDiffRevVector((f1, f2))


    v = AutoDiffRevVector([np.pi/2, 0])
    try:
        np.testing.assert_array_almost_equal(tan(v).val, [2.18504, 0.414214]), 'Tan failed'
    except TypeError:
        print("Caught error as expected")





