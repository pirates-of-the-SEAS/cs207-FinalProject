"""
example

takes as input a function that returns a computational graph
"""
import numpy as np

from .. import AutoDiff, AutoDiffVector
from ..functions import sin


def example_scalar(x):
    x = AutoDiff(name='x', val=x)

    f = sin(x)

    return f

def example_multivariate(x):
    """
    input is an iterable with the correct variable order

    output is an AutoDiffVector and the correct ordering of the variable names
    """
    x1 = AutoDiff(name='x1', val=x[0])
    x2 = AutoDiff(name='x2', val=x[1])
    
    return AutoDiffVector([
        sin(3*x1) - sin(3*x2),
        sin(4*x1) - sin(4*x2)
    ]), ['x1', 'x2']


def __newton_step_scalar(ad):
    val = ad.get_value()
    deriv, _ = ad.get_gradient()
    deriv = deriv[0]

    step = -1 * val/deriv

    return val, deriv, step
    

def __newton_step_multivariate(ad, order):
    val = ad.get_values()
    J, _ = ad.get_jacobian(order=order)

    step = np.linalg.solve(J, -1*val.reshape(-1, 1)).flatten()

    return val, J, step
    

def __determine_scalar_or_vector(x0, f):
    order = None

    # assume x is an iterable
    try:
        try:
            x = x0.flatten()
        except AttributeError:
            x = x0

        results = f(x)

        ad, order = results

        is_vector_func = True

    # if the assumption fails, assume x is a scalar
    except:
        ad = f(x0)

        is_vector_func= False

    return x, ad, is_vector_func, order


def do_newtons_method(x0, f, tol=1e-8, verbose=0):
    """
    x: initial guess (numpy array)
    f: function that returns value and derivative of f at x
    tol: terminate when the absolute value of f at x is less than or equal to the tol

    infers dimensionality of problem based on values returned from f
    """


    x, ad, is_vector_func, order = __determine_scalar_or_vector(x0, f)

    num_iters = 1
    while True:
        if is_vector_func:
            val, J, step = __newton_step_multivariate(ad, order)

            if np.linalg.norm(val, 2) <= tol:
                break

            if verbose > 0:
                print(f"Start of Iteration {num_iters}")
                print("x: {}".format(x))
                print("f(x): {}".format(val))
                print("J: {}".format(J))

            x = x + step

            ad, _ = f(x)

        else:
            val, deriv, step = __newton_step_scalar(ad)

            if np.abs(val) <= tol:
                break

            if verbose > 0:
                print(f"Start of Iteration {num_iters} | x: {x:2f} | f(x): {val:2f} | deriv: {deriv:2f}")

            x = x + step

            ad = f(x)

        num_iters += 1

    if verbose > 0:
        print(f"Converged to {x} after {num_iters} iterations")

    return x

if __name__ == '__main__':
    do_newtons_method(0, example_scalar, tol=1e-8, verbose=0)
