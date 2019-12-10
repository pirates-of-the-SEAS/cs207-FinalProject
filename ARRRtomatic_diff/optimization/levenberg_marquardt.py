"""
Implements Levenberg Marquardt for nonlinear least squares fitting. Module contains
ga collection of example input functions, private helper functions, and the
implementation of Levenberg-Marquardt.
"""
import numpy as np

from .. import AutoDiff, AutoDiffVector
from ..functions import sqrt

def beacon_dist(b1, b2, x, y):
    """
    Helper function used to general nonlinear least squares problem below
    """
    return sqrt((b1 - x)**2 + (b2 - y)**2)


def beacon_resids(b, X, y):
    """Example function to be used in Levenberg-Marquardt routine. 

            INPUTS
            =======
            b: the parameter vector for the nonlinear relationship
            X: the data which is related to y in a nonlinear manner
            y: the response variable
            
            RETURNS
            ========
            rs: an AutoDiffVector containing the residuals
            order: the order of the parameters
        """
    b1 = AutoDiff(name='b1', val=b[0])
    b2 = AutoDiff(name='b2', val=b[1])
    rs = []

    for (x, yy), (d,) in zip(X, y):
        r = d - beacon_dist(b1, b2, x, yy)
        rs.append(r)

    return rs, ['b1', 'b2']


def generate_nonlinear_lsq_data(x_true=0.7, y_true=0.37,
                                   x_data=None, y_data=None):
    """Generates a nonlinear least squares problem.
    Picks an initial starting point in a 2D plane and a bunch of points from
    which we will compute a noisy esimate of their distance to the original point

            INPUTS
            =======
            x_true: the true x value of the point
            y_true: the trye y value of the point
            x_data: an optimal collection of x coordinates
            y_data: an optimal collection of y coordinates

            RETURNS
            ========
            X: the dataset as a numpy array
            d: the distances from each point to the original point
        """

    # Define the beacon locations (randomly located in the unit square)
    if x_data is None:
        x_data = [0.7984, 0.9430, 0.6837, 0.1321, 0.7227, 0.1104, 0.1175, 0.6407,
                0.3288,0.6538]

    if y_data is None:
        y_data = [0.7491, 0.5832, 0.7400, 0.2348, 0.7350, 0.9706, 0.8669,
                  0.0862,0.3664,0.3692]

    # Generate the (noisy) data y, and set initial guess
    std = 0.05
    d = np.zeros(10)
    for i in range(10):
        dx = x_true - x_data[i]
        dy = y_true - y_data[i]
        d[i] = np.sqrt(dx**2 + dy**2) + std*np.random.randn()

    X = np.array([x_data, y_data]).T
    d = np.array(d).reshape(-1,1)

    return X, d


def do_levenberg_marquardt(b0, r, X, y, mu=None, S=None,
                           tol=1e-8, max_iter=2000, verbose=0):
    """
    Performs Levenberg-Marquardy iterations for nonlinear least squares
    fitting given an initial guess, the residual vector, the dataset, and
    algorithm parameters.

    INPUTS
    ======
    b0: the initial guess
    r: a function that produces an AutoDiffVector object containing the
        residuals at the current set of parameters
    X: the covariates that are related to y in a nonlinear manner
    y: the response variable
    mu: the levenberg-marquardt regularization parameter
    S: the levenberg-marquardt regularization matrix
    tol: if the norm of the gradient of sum of squared residuals is smaller
         than this value, the iterations will termiante
    max_iter: the number of iterations to perform levenberg-marquardt
    verbose: the level of verbosity of the output of the routine


    RETURNS
    =======
    b: the parameters which minimize the nonlinear least squares problem

    """

    if S is None:
        S = np.eye(len(b0))

    if mu is None:
        mu = 0.01

    b = b0

    adjustment = mu * np.diag(np.diag(S@S))

    for k in range(max_iter):
        rk, order = beacon_resids(b, X, y)
        adv =  AutoDiffVector(rk)

        r_vals = adv.get_values().reshape(-1,1)
        J, _ = adv.get_jacobian(order)

        rhs = -1 * J.T @ r_vals


        lhs = J.T @ J + adjustment

        if np.linalg.norm(rhs, 2) <= tol:
            print("Converged after {} steps".format(k))
            break

        step = np.linalg.solve(lhs, rhs).flatten()

        b = b + step

    return b


