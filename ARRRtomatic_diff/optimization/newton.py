"""
Implements Newton's Method for a vector valued function. Module contains
a collection of example input functions, private helper functions, and the
implementation of Newton's Method.
"""
import numpy as np

from .. import AutoDiff, AutoDiffVector
from ..functions import sin


def example_scalar(x):
    """Example function for Newton's Method for univariate root finding.

            INPUTS
            =======
            x: a python numeric containing the current guess for the root

            RETURNS
            ========
            f: an AutoDiff object representing the function whose roots are to be found
        """
    x = AutoDiff(name='x', val=x)

    f = sin(x)

    return f

def example_multivariate(x):
    """Example function for Newton's Method for multivariate root finding

            INPUTS
            =======
            x: a 1d numpy array containing the current guess for the root

            RETURNS
            ========
            f: an AutoDiffVector object representing the function whose roots are to be found
            order: the order of the variables
        """
    x1 = AutoDiff(name='x1', val=x[0])
    x2 = AutoDiff(name='x2', val=x[1])
    
    return AutoDiffVector([
        sin(3*x1) - sin(3*x2),
        sin(4*x1) - sin(4*x2)
    ]), ['x1', 'x2']


def __newton_step_scalar(ad):
    "Helper function for performing the newton's method parameter update in the scalar case"""
    val = ad.get_value()
    deriv, _ = ad.get_gradient()
    deriv = deriv[0]

    step = -1 * val/deriv

    return val, deriv, step
    

def __newton_step_multivariate(ad, order):
    """Helper function performing the Newton's method parameter update in the multivariate case"""
    val = ad.get_values()
    J, _ = ad.get_jacobian(order=order)

    step = np.linalg.solve(J, -1*val.reshape(-1, 1)).flatten()

    return val, J, step
    

def __determine_scalar_or_vector(x0, f):
    """
    Helper function for determining whether the input is a scalar or vector

    INPUTS
    ======
    x0: the initial input
    f: the function whose roots are to be determined

    RETURNS
    =======
    x: the initial value 
    ad: the auto diff object
    is_vector_func: the result of the test
    order: the order of the variable names

    """
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
        x = x0

        is_vector_func= False

    return x, ad, is_vector_func, order


def do_newtons_method(x0, f, tol=1e-8, max_iter=2000, verbose=0):
    """
    Performs Newton's Method iterations given an initial guess and function

    INPUTS
    ======
    x0: the initial input
    f: the function whose roots are to be determined. must return either an AutoDiff
        or AutoDiffVector object
    tol: iterations stop when the norm of the vector function is smaller than this value
    max_iter: stop after this # of iterations
    verbose: the level of verbosity when reporting what the routine is doing

    RETURNS
    =======
    x: the root

    """
    x, ad, is_vector_func, order = __determine_scalar_or_vector(x0, f)

    num_iters = 1
    while True:
        if num_iters == max_iter:
            print(f"Did not converge after {max_iter} iterations")
            break

        if is_vector_func:
            val, J, step = __newton_step_multivariate(ad, order)

            if np.linalg.norm(val, 2) <= tol:
                print(f"Converged to {x} after {num_iters} iterations")

                break

            if verbose > 0:
                print(f"Start of Iteration {num_iters}")
                print("x: {}".format(x))
                print("f(x): {}".format(val))
                print("J: \n{}".format(J))

            x = x + step

            ad, _ = f(x)

        else:
            val, deriv, step = __newton_step_scalar(ad)

            if np.abs(val) <= tol:
                if verbose > 0:
                    print(f"Converged to {x} xafter {num_iters} iterations")

                break

            if verbose > 0:
                print(f"Start of Iteration {num_iters} | x: {x:2f} | f(x): {val:2f} | deriv: {deriv:2f}")

            x = x + step

            ad = f(x)

        num_iters += 1

    return x

if __name__ == '__main__':
    do_newtons_method(0, example_scalar, tol=1e-8, verbose=0)
