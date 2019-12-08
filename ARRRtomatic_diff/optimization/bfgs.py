"""
example

takes as input a function that returns a computational graph
"""
import numpy as np

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

    return x**2


def do_bfgs(w0, f, tol=1e-8, max_iter=2000, verbose=0):
    try:
        num_params = len(w0)
        w = w0
    except:
        num_params = 1
        w = np.array([w0])

    B = np.eye(num_params)

    try:
        ad, order = f(w)
    except:
        ad = f(w)
        order = None

    for i in range(max_iter):
        g = ad.get_gradient(order)[0].reshape(-1, 1)

        update = np.linalg.solve(B, -1*g).flatten()

        if np.linalg.norm(update) < 1e-8:
            print("Converged after {} steps".format(i))
            break

        w = w + update

        try:
            ad, _ = f(w)
        except:
            ad = f(w)

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
