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
   f1 = AutoDiffRev(name='x', val=1)
   f2 = AutoDiffRev(name='x', val=3)
   u = AutoDiffRevVector((f1, f2))
   print(u.get_values())
    


