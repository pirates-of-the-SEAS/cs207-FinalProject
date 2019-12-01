"""
Example driver script using Newton's Method to find the roots of sin
"""

from ARRRtomatic_diff import AutoDiff, AutoDiffVector
from ARRRtomatic_diff.functions import sin, exp, sqrt, log
from ARRRtomatic_diff.optimization import (do_newtons_method,
                                           example_scalar,
                                           example_multivariate,
                                           do_bfgs,
                                           rosenbrock,
                                           parabola)



if __name__ == '__main__':
    do_newtons_method(0.2, example_scalar, tol=1e-8, verbose=1)
    do_newtons_method([2, 5.2], example_multivariate, tol=1e-8, verbose=1)

    ans = do_bfgs([-1,1], rosenbrock)
    print(ans)
    ans = do_bfgs([0,1], rosenbrock)
    print(ans)
    ans = do_bfgs([2,1], rosenbrock)
    print(ans)

    ans = do_bfgs(5, parabola)
    print(ans)
   


