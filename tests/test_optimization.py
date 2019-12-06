from ARRRtomatic_diff import AutoDiff, AutoDiffVector
from ARRRtomatic_diff import optimization as opt
import math
import numpy as np
import scipy
import pandas as pd

def test_rosenbrock():
    w = np.array([-1, 1])
    ros, list = opt.rosenbrock(w)
    np.testing.assert_array_equal(ros.val, 4,), 'Rosenbrock function failed'

def test_gradientdescent():
    w0 = np.array([-1, 1])
    output = opt.do_gradient_descent(w0, opt.rosenbrock, max_iter=11000, step_size=0.001)
    np.testing.assert_almost_equal(output, [1, 1], decimal=2), 'Gradient descent failed'
    output = opt.do_gradient_descent(w0, opt.rosenbrock, use_momentum=True, max_iter=2000, step_size=0.001)
    np.testing.assert_almost_equal(output, [1, 1], decimal=3), 'Gradient descent with momentum failed'

def test_example_loss():
    w0 = np.array([-1, 1])
    df = pd.read_csv('../data/sgd_example.csv', header=None).T
    X = df.values
    target, lambdas = opt.example_loss(w0, X, None)
    assert np.allclose(target.val, 0.0004343472)

# def test_sgd():
#     df = pd.read_csv('../data/sgd_example.csv', header=None).T
#     w0 = np.array([-1, 1])
#     X = df.values
#     output = opt.do_stochastic_gradient_descent(w0, opt.rosenbrock, X, tol=1e-6,
#                                                 use_momentum=True,
#                                                 step_size=0.001)
#     np.testing.assert_almost_equal(output, [1, 1], decimal=3), 'Gradient descent failed'

# # def test_