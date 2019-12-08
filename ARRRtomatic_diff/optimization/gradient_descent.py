"""
example

takes as input a function that returns a computational graph
"""

import numpy as np
import scipy
import scipy.optimize
from scipy.optimize import line_search

from .. import AutoDiff, AutoDiffVector
from ..functions import sin, exp

def rosenbrock(w):
    """
    This function serves as a performance test for optimization algorithms.
    The global minimum of Rosenbrock's valley is achieved at (1, 1), where f(x,y) = 0.
    Rosenbrock function is:  f(x,y) = (1 -x)^2 + 100*(y-x^2)^2
    INPUTS
    =======
    w: the initial starting point vector, x0 = w[0], y0 = w[1]

    RETURNS
    ========
    1. f (w)
    2. ordered list of variables x, y
    """
    x = AutoDiff(name='x', val=w[0])
    y = AutoDiff(name='y', val=w[1])

    term1 = 100 * (y - x**2)**2
    term2 = (1-x)**2

    total = term1 + term2

    return total, ['x', 'y']

def parabola(w):
    x = AutoDiff(name='x', val=w[0])

    return x**2, ['x']

def __verify_valid_args(use_line_search,
                        use_momentum,
                        use_adagrad,
                        use_adam,
                        momentum,
                        adam_b1,
                        adam_b2):

    """
    A method to handle the possible set of cases where the inputs to an optimization method may not be valid.
    if inputs are invalid, raises Error.

     INPUTS
    =======
    :param use_line_search: boolean, True or False (whether to use line search)
    :param use_momentum: boolean, True or False
    :param use_adagrad: boolean, True or False (whether to use Adaptive gradient descent)
    :param use_adam: boolean, True or False (whether to use Adaptive momentum estimation)
    :param momentum: the momentum applied to each update
    :param adam_b1: float(scalar), Beta1, which is the momentum decay applied to the first moment estimates
    :param adam_b2: float(scalar, Beta2, which is the exponential decay rate for second moment estimates


    """

    if momentum < 0:
        raise ValueError

    if not (0 < adam_b1 < 1):
        raise ValueError

    if not (0 < adam_b2 < 1):
        raise ValueError

    if (use_line_search + use_momentum + use_adagrad + use_adam) > 1:
        raise Exception("Please use one optimizer at a time")


def __do_line_search_update(get_val, get_gradient, w, direction):
    """
    Performs Armijo line search along the specified "direction"
    starting at point f(w)
    INPUTS
    =======

    :param get_val: callable function f
    :param get_gradient: callable function that calculates grad f
    :param w: initial searching point
    :param direction: line searching direction

    RETURNS
    ========
    optimal step size * direction
    """
    line_search_results = line_search(get_val,
                                      get_gradient,
                                      w,
                                      direction)
    
    step_size = line_search_results[0]

    return step_size * direction

def __do_momentum_update(dw, momentum, step_size, direction):
    dw = momentum * dw + step_size * direction
    return dw

def __do_adagrad_update(G, step_size, grad):
    G += grad.reshape(-1, 1) @ grad.reshape(1, -1)

    diagG = np.diag(G + 0.001)**(-1./2)

    dw = step_size * diagG * -1 * grad

    return dw

# def __do_rmsprop_update(dw, rmsprop, step_size, direction):
#     dw = rmsprop * dw + (1 - rmsprop) * direction**2

#     return step_size * dw**(-1./2) * direction

def __do_adam_update(i, m, v, adam_b1, adam_b2, step_size,
                                  grad, adam_eps):

    m = adam_b1 * m + (1 - adam_b1) * grad
    v = adam_b2 * v + (1 - adam_b2) * grad**2

    mhat = m / (1 - adam_b1**(i+1))
    vhat = v / (1 - adam_b2**(i+1))

    
    return -1 * step_size * mhat * ((np.sqrt(vhat) + adam_eps)**(-1)), m, v
    


def do_gradient_descent(w0, f, tol=1e-8, max_iter=2000, step_size=0.1,
                        verbose=0,
                        use_line_search=False,
                        use_momentum=False,
                        use_adagrad=False,
                        use_adam=False,
                        momentum=0.9,
                        adam_b1=0.9,
                        adam_b2=0.999,
                        adam_eps=0.0001,
                        show=False # plz don't erase
                        ):

    __verify_valid_args(use_line_search,
                        use_momentum,
                        use_adagrad,
                        use_adam,
                        momentum,
                        adam_b1,
                        adam_b2)


    try:
        num_params = len(w0)
        w = w0
    except:
        num_params = 1
        w = np.array([w0])
    if show:
        w_path = []
        w_path.append(w)

    def get_val(w):
        try:
            ad, _ = f(w)
        except TypeError:
            ad = f(w)

        return ad.get_value()

    def get_gradient(w):
        try:
            ad, order = f(w)
        except TypeError:
            ad = f(w)
            order = None

        return ad.get_gradient(order)[0]

    for i in range(max_iter):
        grad = get_gradient(w)

        if np.linalg.norm(grad) <= tol:
            print("Converged after {} steps".format(i))
            break

        direction = -1 * grad

        if use_line_search:
            dw = __do_line_search_update(get_val, get_gradient, w, direction)

        if use_momentum:
            if i == 0:
                dw = 0

            dw = __do_momentum_update(dw, momentum, step_size, direction)

        if use_adagrad:
            if i == 0:
                G = np.zeros((num_params, num_params))

            dw = __do_adagrad_update(G, step_size, grad)

        # if use_rmsprop:
        #     if i == 0:
        #         dw = 0

        #     dw = __do_rmsprop_update(dw, rmsprop, step_size, direction)

        if use_adam:
            if i == 0:
                m = 0
                v = 0

            dw, m, v = __do_adam_update(i, m, v, adam_b1, adam_b2, step_size,
                                  grad, adam_eps)

        if not any([use_line_search, use_momentum, use_adagrad,
                    use_adam]):
            dw = step_size * direction
        # print(w)
        w = w + dw
        if show:
            w_path.append(w)
    else:
        print(f"Did not converge after {max_iter} steps")

    if show:
        return w_path

    return w

def example_loss(params, X_data, y_data):
    a = 0.000045
    b = -0.000098
    c = 0.003926

    lambda1 = AutoDiff(name='lambda1', val=params[0])
    lambda2 = AutoDiff(name='lambda2', val=params[1])

    total = 0
    size = len(X_data)

    for (x,y) in X_data:
        term1 = a * lambda2**2 * y
        term2 = b * lambda1**2 * x
        term3 = c * lambda1 * x * exp((y**2 - x**2) * (lambda1**2 + lambda2**2))
    
        total += term1 + term2 + term3
    
    return total/size, ['lambda1', 'lambda2']


def do_stochastic_gradient_descent(w0,
                                   f,
                                   X,
                                   y=None,
                                   num_epochs=100,
                                   batch_size=64,
                                   step_size=0.1,
                                   tol=1e-8,
                                   verbose=0,
                                   use_momentum=False,
                                   use_adagrad=False,
                                   use_adam=False,
                                   momentum=0.9,
                                   adam_b1=0.9,
                                   adam_b2=0.999,
                                   adam_eps=0.0001):

    __verify_valid_args(False,
                        use_momentum,
                        use_adagrad,
                        use_adam,
                        momentum,
                        adam_b1,
                        adam_b2)

    X_orig = X.copy()
    if y is not None:
        y_orig = y.copy()
    else:
        y_orig = None

    num_data = len(X_orig)

    try:
        num_params = len(w0)
        w = w0
    except:
        num_params = 1
        w = np.array([w0])

    for i in range(num_epochs):
        L, order = f(w, X_orig, y_orig)

        if np.linalg.norm(L.get_gradient(order)[0], 2) <= tol:
            print("Converged after {} epochs".format(i))
            break


        print("Loss at start of epoch {}: {}".format(i, L.get_value()))
        print(i)

        if verbose > 0:
            print("params:")
            print(w)


        idx = np.random.permutation(num_data)
        num_iters = int(num_data // batch_size)
        for j in range(num_iters):
            start_idx = j * batch_size
            end_idx = min((j+1) * batch_size, num_data - 1)

            batch_idx = idx[start_idx: end_idx]

            batch_X = X_orig[batch_idx]

            if y is not None:
                batch_y = y_orig[batch_idx]
            else:
                batch_y = None


            ad, _ = f(w, batch_X, batch_y)

            grad, _ = ad.get_gradient(order)

            direction = -1 * grad


            if use_momentum:
                if i == 0:
                    dw = 0

                dw = __do_momentum_update(dw, momentum, step_size, direction)

            if use_adagrad:
                if i == 0:
                    G = np.zeros((num_params, num_params))

                dw = __do_adagrad_update(G, step_size, grad)

            # if use_rmsprop:
            #     if i == 0:
            #         dw = 0

            #     dw = __do_rmsprop_update(dw, rmsprop, step_size, direction)

            if use_adam:
                if i == 0:
                    m = 0
                    v = 0

                dw, m, v = __do_adam_update(i, m, v, adam_b1, adam_b2, step_size,
                                      grad, adam_eps)

            if not any([use_momentum, use_adagrad, use_adam]):
                dw = step_size * direction

            w = w + dw

    else:
        print(f"Did not converge after {num_epochs} epochs")
        
    return w


if __name__ == '__main__':
    do_gradient_descent(0, example_scalar, tol=1e-8, verbose=0)

