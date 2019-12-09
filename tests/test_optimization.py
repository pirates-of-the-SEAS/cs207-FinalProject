from ARRRtomatic_diff import AutoDiff, AutoDiffVector
from ARRRtomatic_diff import optimization as opt
from ARRRtomatic_diff.optimization.gradient_descent import rosenbrock as rb
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
    output = opt.do_gradient_descent(w0, rb, max_iter=11000, step_size=0.001)
    np.testing.assert_almost_equal(output, [1, 1], decimal=2), 'Gradient descent failed'
    output = opt.do_gradient_descent(w0, opt.rosenbrock, use_momentum=True, max_iter=2000, step_size=0.001)
    np.testing.assert_almost_equal(output, [1, 1], decimal=3), 'Gradient descent with momentum failed'

def test_example_loss():
    w0 = np.array([-1, 1])
    df = pd.read_csv('./data/sgd_example.csv', header=None).T
    # df = pd.read_csv('../data/sgd_example.csv', header=None).T
    X = df.values
    target, lambdas = opt.example_loss(w0, X, None)
    assert np.allclose(target.val, 0.0004343472)

# def test_sgd():
#     w0 = np.array([0, 1])
#     df = pd.read_csv('./data/sgd_example.csv', header=None).T
#     # df = pd.read_csv('../data/sgd_example.csv', header=None).T
#     X = df.values
#     output = opt.do_stochastic_gradient_descent(w0, opt.example_loss, X, num_epochs=25, tol=1e-6,
#                                                 use_momentum=True,
#                                                 use_adagrad=False,
#                                                 use_adam=False,
#                                                 step_size=0.28) #0.2 works and 40 epochs
#     np.testing.assert_almost_equal(output, [2.054, 0.04], decimal=2), 'Stochastic gradient descent failed'
#

### IS THE DESCRIPTION OF THIS FUNCTION CORRECT IN THE optimization FOLDER????
### SEEMS THAT F MAY NOT RETURN VALUE AND &&&& AND &&& DERIVATIVE ???
def test_newtons_method_scalar():
    result = opt.do_newtons_method(1, opt.example_scalar, tol=1e-8, verbose=0)
    np.testing.assert_almost_equal(result, 0), "Newton's method for the scalar case failed"

def test_newtons_method_scalar_verbose():
    result = opt.do_newtons_method(1, opt.example_scalar, tol=1e-8, verbose=1)
    np.testing.assert_almost_equal(result, 0), "Newton's method for the scalar case failed"

def test_newtons_method_scalar_maxedout():
    result = opt.do_newtons_method(1.1, opt.example_scalar, tol=1e-8, max_iter=6, verbose=0)
    np.testing.assert_almost_equal(result, 0), "Newton's method for the scalar case failed"

def test_newtons_method_vector():
    w0 = np.array([1, 0])
    result = opt.do_newtons_method(w0, opt.example_multivariate, tol=1e-8, verbose=0)
    np.testing.assert_almost_equal(result, [10.7337749, -9.6865773]), "Newton's method for the vector case failed"

def test_newtons_method_vector_verbose():
    w0 = np.array([1, 0])
    result = opt.do_newtons_method(w0, opt.example_multivariate, tol=1e-8, verbose=1)
    np.testing.assert_almost_equal(result, [10.7337749, -9.6865773]), "Newton's method for the vector case failed"

def test_newtons_method_junk():
    try:
        result = opt.do_newtons_method("junk", opt.example_scalar, tol=1e-8, verbose=0)
    except TypeError:
        print("Caught error as expected")

def test_bfgs_scalar():
    bfgs_output = opt.do_bfgs(0, opt.example_scalar, tol=1e-8, verbose=0)
    assert np.allclose(bfgs_output, -1.5707963), 'BFGS scalar failed'

def test_bfgs_scalar_parabola():
    bfgs_output = opt.do_bfgs(-953.4, opt.parabola, tol=1e-8, max_iter=2000, verbose=0)
    assert np.allclose(bfgs_output, 0), 'BFGS scalar failed'

def test_bfgs_scalar_nonconverge():
    bfgs_output = opt.do_bfgs(0.8, opt.example_scalar, tol=1e-8, max_iter=3, verbose=1)
    assert np.allclose(bfgs_output, 1.42417872), 'BFGS scalar failed'

def test_bfgs_vector():
    w0 = np.array([-1, 1])
    bfgs_output = opt.do_bfgs(w0, opt.rosenbrock)
    np.testing.assert_almost_equal(bfgs_output, [1, 1]), 'BFGS vector failed'

def test_levenberg_marquardt():
    X, y = opt.generate_nonlinear_lsq_data()
    b0 = np.array([0.5, 0.5])
    r = opt.beacon_resids(b0, X, y)
    result_lm = opt.do_levenberg_marquardt(b0, r, X, y, tol=1e-10, max_iter=6000)
    np.testing.assert_almost_equal(result_lm, [0.7, 0.4], decimal=1), 'Levenberg-Marquardt failed'

def test_levenberg_marquardt_verbose():
    X, y = opt.generate_nonlinear_lsq_data()
    b0 = np.array([0.5, 0.5])
    r = opt.beacon_resids(b0, X, y)
    result_lm = opt.do_levenberg_marquardt(b0, r, X, y, tol=1e-10, max_iter=6000, verbose=1)
    np.testing.assert_almost_equal(result_lm, [0.744902, 0.362923], decimal=1), 'Levenberg-Marquardt failed'

def test_levenberg_marquardt_junk():
    X, y = opt.generate_nonlinear_lsq_data()
    b0 = np.array(["dfsd", 0.5])
    try:
        r = opt.beacon_resids(b0, X, y)
        result_lm = opt.do_levenberg_marquardt(b0, r, X, y, tol=1e-10, max_iter=6000, verbose=1)
        np.testing.assert_almost_equal(result_lm, [0.744902, 0.362923], decimal=1), 'Levenberg-Marquardt failed'
    except TypeError:
        print("Caught error as expected")