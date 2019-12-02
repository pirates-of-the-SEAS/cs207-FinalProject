"""
example

takes as input a function that returns a computational graph
"""

import numpy as np
from scipy.optimize import line_search

from .. import AutoDiff, AutoDiffVector
from ..functions import sin

def rosenbrock(w):
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
                        use_rmsprop,
                        use_adam,
                        momentum,
                        rmsprop,
                        adam_b1,
                        adam_b2):

    if momentum < 0:
        raise ValueError

    if rmsprop < 0:
        raise ValueError

    if not (0 < adam_b1 < 1):
        raise ValueError

    if not (0 < adam_b2 < 1):
        raise ValueError

    if (use_line_search + use_momentum + use_adagrad + use_rmsprop + use_adam) > 1:
        raise Exception("Please use one optimizer at a time")


def __do_line_search_update(get_val, get_gradient, w, direction):
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

    diagG = np.diag(G)**(-1./2)

    dw = step_size * diagG * -1 * grad

    return dw

def __do_rmsprop_update(dw, rmsprop, step_size, direction):
    dw = rmsprop * dw + (1 - rmsprop) * direction**2


    return step_size * dw**(-1./2) * direction

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
                        use_rmsprop=False,
                        use_adam=False,
                        momentum=0.9,
                        rmsprop=0.9,
                        adam_b1=0.9,
                        adam_b2=0.999,
                        adam_eps=0.0001):

    __verify_valid_args(use_line_search,
                        use_momentum,
                        use_adagrad,
                        use_rmsprop,
                        use_adam,
                        momentum,
                        rmsprop,
                        adam_b1,
                        adam_b2)

    try:
        num_params = len(w0)
        w = w0
    except:
        num_params = 1
        w = np.array([w0])

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

        if use_rmsprop:
            if i == 0:
                dw = 0

            dw = __do_rmsprop_update(dw, rmsprop, step_size, direction)

        if use_adam:
            if i == 0:
                m = 0
                v = 0

            dw, m, v = __do_adam_update(i, m, v, adam_b1, adam_b2, step_size,
                                  grad, adam_eps)

        w = w + dw
    else:
        print(f"Did not converge after {max_iter} steps")
        
    return w


if __name__ == '__main__':
    do_gradient_descent(0, example_scalar, tol=1e-8, verbose=0)

