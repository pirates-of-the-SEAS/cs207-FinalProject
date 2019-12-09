"""
Example driver script using Newton's Method to find the roots of sin

Note + iterable + autodiff yields collection of autodiffs.
have to make autodiffvector

"""

import pandas as pd
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
                                           do_levenberg_marquardt,
                                           example_loss,
                                           do_stochastic_gradient_descent

)



if __name__ == '__main__':
    X, d = generate_nonlinear_lsq_data(x_true=0.7, y_true=0.37,
                                   x_data=None, y_data=None)

    
    w = do_levenberg_marquardt([0.6, 0.4], beacon_resids, X, d,
                               mu=None,
                               S=None,
                               tol=1e-8, max_iter=2000, verbose=0)

    print(w)


    # df = pd.read_csv('./data/sgd_example.csv', header=None).T
    # df.columns = ['x', 'y']
    # assert(np.allclose(df['y']**2 - df['x']**2, -0.1))

    # lambda1 = 2
    # lambda2 = 1

    # X = df.values

    # do_stochastic_gradient_descent(np.array([lambda1, lambda2]),
    #                                example_loss,
    #                                X,
    #                                y=None,
    #                                num_epochs=100,
    #                                batch_size=64,
    #                                step_size=0.1,
    #                                verbose=1,
    #                                use_momentum=False,
    #                                use_adagrad=False,
    #                                use_adam=True)

    



  

