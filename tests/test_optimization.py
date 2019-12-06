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
    df = pd.read_csv('./data/sgd_example.csv', header=None).T
    X = df.values
    target, lambdas = opt.example_loss(w0, X, None)
    assert np.allclose(target.val, 0.0004343472)

def test_sgd():
    w0 = np.array([-1, 1])
    df = pd.read_csv('./data/sgd_example.csv', header=None).T
    X = df.values
    output = opt.do_stochastic_gradient_descent(w0, opt.example_loss, X, num_epochs=100, tol=1e-6,
                                                use_momentum=True,
                                                step_size=0.1)
    np.testing.assert_almost_equal(output, [2.054, 0.04], decimal=2), 'Stochastic gradient descent failed'


### IS THE DESCRIPTION OF THIS FUNCTION CORRECT IN THE optimization FOLDER????
### SEEMS THAT F MAY NOT RETURN VALUE AND &&&& AND &&& DERIVATIVE ???
def test_newtons_method_scalar():
    result = opt.do_newtons_method(0, opt.example_scalar, tol=1e-8, verbose=0)
    np.testing.assert_almost_equal(result, 0), "Newton's method for the scalar case failed"

def test_newtons_method_multivariate():
    w0 = np.array([1, 0])
    result = opt.do_newtons_method(w0, opt.example_multivariate, tol=1e-8, verbose=0)
    np.testing.assert_almost_equal(result, [10.7337749, -9.6865773]), "Newton's method for the vector case failed"

def test_bfgs_scalar():
    bfgs_output = opt.do_bfgs(0, opt.example_scalar, tol=1e-8, verbose=0)
    print("RESULT", bfgs_output)
    assert np.allclose(bfgs_output, -1.5707963), 'BFGS scalar failed'

# def