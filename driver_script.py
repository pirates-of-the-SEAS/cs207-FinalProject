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
                                           parabola,
                                           do_gradient_descent)



if __name__ == '__main__':
    ans = do_gradient_descent([0,1],
                              rosenbrock,
                              use_line_search=False,
                              step_size=0.01,
                              use_momentum=False,
                              use_adagrad=False,
                              use_rmsprop=False,
                              use_adam=True,
                              max_iter=10000)

    print(ans)

    # ans = do_gradient_descent(5, parabola, use_line_search=True)
    # print(ans)
   


