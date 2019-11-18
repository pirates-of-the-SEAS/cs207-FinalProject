"""
Example driver script using Newton's Method to find the roots of sin
"""

from ARRRtomatic_diff import AutoDiff
from ARRRtomatic_diff.functions import sin, exp, sqrt, log

x1 = AutoDiff(name='x', val=0)
x2 = AutoDiff(name='x', val=3)
y = AutoDiff(name='y', val=4)

log(y)



print(x2 + x2 + x1)

def f(x):
    x = AutoDiff(name='x', val=x)

    auto_diff_results = sin(x)

    return auto_diff_results['val'], auto_diff_results['d_x']

def do_newtons_method(x, f, tol=1e-6, verbose=0):
    """
    x: initial guess
    f: function that returns value and derivative of f at x
    tol: terminate when the absolute value of f at x is less than or equal to the tol
    """
    num_iters = 1
    while abs(f(x)[0]) > tol:
        val, deriv = f(x)

        if verbose > 0:
            print(f"Iteration {num_iters} | x: {x:2f} | f(x): {val:2f} | deriv: {deriv:2f}")

        x = x - val/deriv

        

        num_iters += 1

    if verbose > 0:
        print(f"Converged to {x} after {num_iters} iterations")

    return x


if __name__ == '__main__':
    do_newtons_method(0.2, f, verbose=1)
    do_newtons_method(0.8, f, verbose=1)
    do_newtons_method(1.2, f, verbose=1)
    do_newtons_method(1.8, f, verbose=1)
    do_newtons_method(2.2, f, verbose=1)
