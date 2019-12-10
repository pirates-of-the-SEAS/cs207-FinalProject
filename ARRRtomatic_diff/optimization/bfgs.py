"""
Implements BFGS for a function with multivariate inputs. Module contains
a collection of example input functions, private helper functions, and the
implementation of BFGS.
"""
import numpy as np

from .. import AutoDiff, AutoDiffVector
from ..functions import sin

def rosenbrock(w):
    """Example function for optimization

            INPUTS
            =======
            w: a python numeric containing the current guess for the minimum

            RETURNS
            ========
            f: an AutoDiff object representing the optimization objective
            order: a list containing the order of the variable names
        """
    x = AutoDiff(name='x', val=w[0])
    y = AutoDiff(name='y', val=w[1])

    term1 = 100 * (y - x**2)**2
    term2 = (1-x)**2

    total = term1 + term2

    return total, ['x', 'y']

def parabola(w):
    """Example function for optimization

            INPUTS
            =======
            w: a python numeric containing the current guess for the minimum

            RETURNS
            ========
            f: an AutoDiff object representing the optimization objective
            order: a list containing the order of the variable names
        """
    x = AutoDiff(name='x', val=w[0])

    return x**2


def do_bfgs(w0, f, tol=1e-8, max_iter=2000, verbose=0):
    """
    Performs BFGS iterations given an initial guess and function

    INPUTS
    ======
    x0: the initial input
    f: the function whose minimum will be sought. must return either an AutoDiff
        or AutoDiffVector object
    tol: iterations stop when the norm of the vector function is smaller than this value
    max_iter: stop after this # of iterations
    verbose: the level of verbosity when reporting what the routine is doing

    RETURNS
    =======
    x: the guess for the minimum

    """

    # determine whether function is scalar or iterable
    try:
        num_params = len(w0)
        w = w0
    except:
        num_params = 1
        w = np.array([w0])

    # initialize BFGS matrix
    B = np.eye(num_params)

    try:
        ad, order = f(w)
    except:
        ad = f(w)
        order = None

    for i in range(max_iter):
        g = ad.get_gradient(order)[0].reshape(-1, 1)

        # get bfgs update
        update = np.linalg.solve(B, -1*g).flatten()

        if np.linalg.norm(update) < 1e-8:
            print("Converged after {} steps".format(i))
            break

        w = w + update

        try:
            ad, _ = f(w)
        except:
            ad = f(w)

        # update Hessian approximation
        y = ad.get_gradient(order)[0].reshape(-1, 1) - g
        s = update.reshape(-1, 1)

        term1 = (y @ y.T) / (y.T @ s)
        term2 = (B @ s @ s.T @ B) / (s.T @ B @ s)

        deltab = term1 - term2

        B = B + deltab

    else:
        print("Did not converge after 2000 steps")
        
    return w




if __name__ == '__main__':
    do_bfgs(0, example_scalar, tol=1e-8, verbose=0)
