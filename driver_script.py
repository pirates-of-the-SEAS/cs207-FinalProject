"""
Example driver script using Newton's Method to find the roots of sin
"""

from ARRRtomatic_diff import AutoDiff, AutoDiffVector
from ARRRtomatic_diff.functions import sin, exp, sqrt, log
from ARRRtomatic_diff.optimization import (do_newtons_method, example_scalar,
                                           example_multivariate)



if __name__ == '__main__':
    do_newtons_method(0.2, example_scalar, tol=1e-8, verbose=1)
    do_newtons_method([0.26, 5], example_multivariate, tol=1e-8, verbose=1)
   


