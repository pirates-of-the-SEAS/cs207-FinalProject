from ARRRtomatic_diff import AutoDiff, AutoDiffVector
from ARRRtomatic_diff import functions as ad
import numpy as np


def test_sin():
    f1 = AutoDiff(name='x', val=0)
    f2 = AutoDiff(name='y', val=np.pi/2)
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector([0, np.pi/2])
    np.testing.assert_array_almost_equal(ad.sin(u).val, [0, 1]), 'Sine failed'
    J, order = (ad.sin(u)).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[1, 0], [0, 0]]), 'Sine failed'

    # np.testing.assert_array_almost_equal(ad.sin(v).val, [0, 1]), 'Sine failed'
    # J, order = (ad.sin(v)).get_jacobian()
    # np.testing.assert_array_almost_equal(J, [[1, 0], [0, 0]]), 'Sine failed'


def test_cos():
    f1 = AutoDiff(name='x', val=0)
    f2 = AutoDiff(name='y', val=np.pi/2)
    u = AutoDiffVector((f1, f2))
    np.testing.assert_array_almost_equal(ad.cos(u).val, [1, 0]), 'Cosine failed'
    J, order = (ad.cos(u)).get_jacobian()
    np.testing.assert_array_equal(J, [[0, 0], [0, -1]]), 'Cosine failed'


def test_tan():
    f1 = AutoDiff(name='x', val=-2)
    f2 = AutoDiff(name='y', val=np.pi/8)
    u = AutoDiffVector((f1, f2))
    np.testing.assert_array_almost_equal(ad.tan(u).val, [2.18504, 0.414214]), 'Tan failed'
    J, order = (ad.tan(u)).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[5.774399, 0], [0, 1.171573]], decimal=4), 'Tan failed'
    v = AutoDiffVector([np.pi/2, 0])
    try:
        np.testing.assert_array_almost_equal(ad.tan(v).val, [2.18504, 0.414214]), 'Tan failed'
    except TypeError:
        print("Caught error as expected")


def test_csc():
    f1 = AutoDiff(name='x', val=-2)
    f2 = AutoDiff(name='y', val=np.pi / 8)
    u = AutoDiffVector((f1, f2))
    np.testing.assert_array_almost_equal(ad.csc(u).val, [-1.09975, 2.613126]), 'Cosecant failed'
    J, order = (ad.csc(u)).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[0.5033, 0], [0, -6.3086]], decimal=4), 'Cosecant failed'


def test_sec():
    f1 = AutoDiff(name='x', val=0)
    f2 = AutoDiff(name='y', val=np.pi)
    u = AutoDiffVector((f1, f2))
    np.testing.assert_array_almost_equal(ad.sec(u).val, [1, -1]), 'Secant failed'
    J, order = (ad.sec(u)).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[0, 0], [0, 0]], decimal=4), 'Secant failed'


def test_cot():
    f1 = AutoDiff(name='x', val=4)
    f2 = AutoDiff(name='y', val=np.pi/8)
    u = AutoDiffVector((f1, f2))
    np.testing.assert_array_almost_equal(ad.cot(u).val, [0.863691, 2.414214]), 'Cotangent failed'
    J, order = (ad.cot(u)).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[-1.746, 0], [0, -6.8284]], decimal=4), 'Cotangent failed'
