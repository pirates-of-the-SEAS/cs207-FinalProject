from ARRRtomatic_diff import optimization as opt
from ARRRtomatic_diff.optimization.gradient_descent import rosenbrock as rb
from ARRRtomatic_diff.optimization.gradient_descent import parabola as pb
import random
import numpy as np
import pandas as pd


def test_rosenbrock():
    w = np.array([-1, 1])
    ros, list = opt.rosenbrock(w)
    np.testing.assert_array_equal(ros.val, 4,), 'Rosenbrock function failed'


def test_gradientdescent_scalar():
    w0 = -87
    w2 = 5
    output = opt.do_gradient_descent(w0, pb, max_iter=1000, step_size=0.01)
    np.testing.assert_almost_equal(output, 0, decimal=2), 'Gradient descent failed'
    output = opt.do_gradient_descent(w0, pb, use_momentum=True, max_iter=2000, step_size=0.001)
    np.testing.assert_almost_equal(output, 0, decimal=3), 'Gradient descent with momentum failed'
    output = opt.do_gradient_descent(w0, pb, use_line_search=True, max_iter=2000, step_size=0.001)
    np.testing.assert_almost_equal(output, 0, decimal=3), 'Gradient descent with line search failed'
    output = opt.do_gradient_descent(w0, pb, use_adagrad=True, max_iter=3000, step_size=10)
    np.testing.assert_almost_equal(output, 0, decimal=3), 'Gradient descent with adagrad failed'
    output = opt.do_gradient_descent(w0, pb, use_adam=True, max_iter=2000, step_size=19)
    np.testing.assert_almost_equal(output, 0, decimal=3), 'Gradient descent with adam failed'
    try:
        opt.do_gradient_descent(w0, pb, use_momentum=True, max_iter=2000, step_size=0.001)
    except ValueError:
        print("Caught error as expected")
    try:
        opt.do_gradient_descent(w0, pb, momentum=-3, max_iter=2000, step_size=0.001)
    except ValueError:
        print("Caught error as expected")
    try:
        opt.do_gradient_descent(w0, pb, adam_b1=50, max_iter=2000, step_size=0.001)
    except ValueError:
        print("Caught error as expected")
    try:
        opt.do_gradient_descent(w0, pb, adam_b2=0, max_iter=2000, step_size=0.001)
    except ValueError:
        print("Caught error as expected")
    try:
        opt.do_gradient_descent(w0, pb, use_line_search=True, use_momentum=True, step_size=0.001)
    except Exception:
        print("Caught error as expected")


def test_gradientdescent_vector():
    w0 = np.array([-1, 1])
    output = opt.do_gradient_descent(w0, rb, max_iter=11000, step_size=0.001)
    np.testing.assert_almost_equal(output, [1, 1], decimal=2), 'Gradient descent failed'
    output = opt.do_gradient_descent(w0, rb, use_momentum=True, max_iter=2000, step_size=0.001)
    np.testing.assert_almost_equal(output, [1, 1], decimal=3), 'Gradient descent with momentum failed'


def test_example_loss():
    w0 = np.array([-1, 1])
    df = pd.read_csv('./data/sgd_example.csv', header=None).T
    # df = pd.read_csv('../data/sgd_example.csv', header=None).T
    X = df.values
    target, lambdas = opt.example_loss(w0, X, None)
    assert np.allclose(target.val, 0.0004343472)


def test_sgd():
    w0 = np.array([0, 1])
    np.random.seed(1)
    random.seed(1)
    df = pd.read_csv('./data/sgd_example.csv', header=None).T
    # df = pd.read_csv('../data/sgd_example.csv', header=None).T
    X = df.values
    output = opt.do_stochastic_gradient_descent(w0, opt.example_loss, X, num_epochs=20, tol=1e-6,
                                                use_momentum=True,
                                                use_adagrad=False,
                                                use_adam=False,
                                                step_size=0.28)
    np.testing.assert_almost_equal(output, [2.053, 0.117], decimal=3), 'Stochastic gradient descent with momentum failed'


def test_sgd_adagrad():
    w0 = np.array([0, 0])
    np.random.seed(10)
    random.seed(10)
    df = pd.read_csv('./data/sgd_example.csv', header=None).T
    # df = pd.read_csv('../data/sgd_example.csv', header=None).T
    X = df.values
    output = opt.do_stochastic_gradient_descent(w0, opt.example_loss, X, num_epochs=30, tol=1e-6,
                                                use_momentum=False,
                                                use_adagrad=True,
                                                use_adam=False,
                                                step_size=0.58)
    np.testing.assert_almost_equal(output, [-2.479,  0], decimal=3), 'Stochastic gradient descent with adagrad failed'


def test_sgd_adam():
    w0 = np.array([0, 1])
    np.random.seed(10)
    random.seed(10)
    df = pd.read_csv('./data/sgd_example.csv', header=None).T
    # df = pd.read_csv('../data/sgd_example.csv', header=None).T
    X = df.values
    output = opt.do_stochastic_gradient_descent(w0, opt.example_loss, X, num_epochs=25, tol=1e-6,
                                                verbose=1,
                                                use_momentum=False,
                                                use_adagrad=False,
                                                use_adam=True,
                                                step_size=0.48)
    np.testing.assert_almost_equal(output, [2.05e+00, -2.01e-03], decimal=2), 'Stochastic gradient descent with adam failed'


def test_sgd_unaltered():
    w0 = np.array([0, 1])
    np.random.seed(1)
    random.seed(1)
    df = pd.read_csv('./data/sgd_example.csv', header=None).T
    # df = pd.read_csv('../data/sgd_example.csv', header=None).T
    X = df.values
    output = opt.do_stochastic_gradient_descent(w0, opt.example_loss, X, num_epochs=1, tol=1e-6,
                                                use_momentum=False,
                                                use_adagrad=False,
                                                use_adam=False,
                                                step_size=0.28)
    np.testing.assert_almost_equal(output, [0.031, 0.9997], decimal=1), 'Stochastic gradient descent with momentum failed'


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