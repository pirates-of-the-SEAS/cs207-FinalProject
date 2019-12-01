"""
Example driver script using Newton's Method to find the roots of sin
"""

from ARRRtomatic_diff import AutoDiff, AutoDiffVector
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
    x = AutoDiff(name='x', val=1)
    y = AutoDiff(name='y', val=3)
    z = AutoDiff(name='z', val=20)

    u = AutoDiffVector((
        x,
        y
    ))

    v = AutoDiffVector((
        z,
        z
    ))

    print(u)
    print(x.get_gradient())
    print(y.get_gradient())

    print(u.get_values())
    print(u.get_jacobian())
    print(u.dot(u))

    a = u + v

    
    print(a.get_values())

    # # # performs vector addition, scalar multiplication, and broadcasts the 
    # # # unary operator sin element-wise
    # z = sin(5*(u + v))

    # # # a numpy array representnug the Jacobian of the vector-valued function
    # # # f1 = x - y
    # # # f2 = y + x
    # J = z.get_jacobian()

    # z2 = z*x

    # J2 = z2.get_jacobian()

    # print(J)
    # print(J2)


