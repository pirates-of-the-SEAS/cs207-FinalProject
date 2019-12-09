from ARRRtomatic_diff import AutoDiffVector, AutoDiff
from ARRRtomatic_diff import functions as ad
import numpy as np


def test_sqrt():
    f1 = AutoDiff(name='x', val=16)
    f2 = AutoDiff(name='y', val=64)
    f3 = AutoDiff(name='z', val=-1)
    u = AutoDiffVector((f1, f2))
    u2 = AutoDiffVector((f1, f3))
    v = AutoDiffVector([16, 64])
    t = AutoDiffVector([0, 4])
    np.testing.assert_array_equal(ad.sqrt(u), [4, 8]), 'Square root failed'
    J, order = (ad.sqrt(u)).get_jacobian()
    np.testing.assert_array_equal(J, [[0.125, 0], [0, 0.0625]]), 'Square root failed'
    np.testing.assert_array_equal(ad.sqrt(v), [4, 8]), 'Square root failed'
    J, order = (ad.sqrt(v)).get_jacobian()
    np.testing.assert_array_equal(J, [[0], [0]]), 'Square root failed'
    np.testing.assert_array_equal(ad.sqrt(t), [0, 2]), 'Square root failed'
    try:
        ad.sqrt(u2)
    except ValueError:
        print("Caught error as expected")


def test_euler():
    f1 = AutoDiff(name='x', val=0)
    f2 = AutoDiff(name='y', val=3)
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector((-f1, -f2))
    np.testing.assert_array_equal(ad.exp(u), [1, 20.085536923187664]), "Euler's number failed"
    J, order = (ad.exp(u)).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[1, 0], [0, 20.085537]]), "Euler's number failed"
    np.testing.assert_array_almost_equal(ad.exp(v), [1, 0.04978706836786395]), "Euler's number failed"
    J, order = (ad.exp(v)).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[-1, 0], [0, -0.049787]]), "Euler's number failed"


def test_log():
    f1 = AutoDiff(name='x', val=1)
    f2 = AutoDiff(name='y', val=3)
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector((-f1, -f2))
    np.testing.assert_array_equal(ad.log(u), [np.log(1), np.log(3)]), "Log failed"
    J, order = (ad.log(u)).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[1, 0], [0,  0.333333]]), "Log failed"
    try:
        np.testing.assert_array_almost_equal(ad.log(v), [1, 0.04978706836786395])
    except ValueError:
        print("Caught error as expected ")


def test_composition():
    f1 = AutoDiff(name='x', val=5)
    f2 = AutoDiff(name='y', val=3)
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector((-f2, -f2))
    z = (u + v) * u * (1/v)
    np.testing.assert_array_almost_equal(z, [(-10/3), 0]), "Composition failed"
    J, order = (z.get_jacobian())
    np.testing.assert_array_almost_equal(J, [[-2.333333, 2.777778], [0, 0]]), "Composition failed"
    np.testing.assert_array_almost_equal(ad.exp(u)**2, [22026.465794806707, 403.428793492735])
    J, order = (ad.exp(u)**2).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[44052.93158961341, 0], [0, 806.85758698547]]), "Composition failed"