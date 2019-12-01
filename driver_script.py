"""
Example driver script using Newton's Method to find the roots of sin
"""

from ARRRtomatic_diff import AutoDiff
from ARRRtomatic_diff.functions import sin, exp, sqrt, log


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
    # do_newtons_method(0.2, f, verbose=1)
    # do_newtons_method(0.8, f, verbose=1)
    # do_newtons_method(1.2, f, verbose=1)
    # do_newtons_method(1.8, f, verbose=1)
    # do_newtons_method(2.2, f, verbose=1)

    x1 = AutoDiff(name='x', val=1)
    x2 = AutoDiff(name='x', val=1)
    x3 = AutoDiff(name='x', val=2)
    y = AutoDiff(name='y', val=2)

    print(x1)
    print(y)
    z1 = x1 + y
    z2 = x1 + x2
    print(z1)
    print(x1 + x2)

    try:
        z3 = x1 + x3
    except:
        print("Caught expected exception")

    try:
        z3 = z1 + x3
    except:
        print("Caught expected exception")

    print(z1 + x2)

    print(z1)

    print(z1.get_named_variables())

