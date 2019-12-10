from ARRRtomatic_diff import AutoDiff, AutoDiffVector
from ARRRtomatic_diff import functions as ad
import numpy as np


def test_sin():
    f1 = AutoDiff(name='x', val=0)
    f2 = AutoDiff(name='y', val=np.pi/2)
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector([0, np.pi/2])
    np.testing.assert_array_almost_equal(ad.sin(u), [0, 1]), 'Sine failed'
    J, order = (ad.sin(u)).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[1, 0], [0, 0]]), 'Sine failed'
    np.testing.assert_array_almost_equal(ad.sin(v), [0, 1]), 'Sine failed'
    J, order = (ad.sin(v)).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[0], [0]]), 'Sine failed'


def test_cos():
    f1 = AutoDiff(name='x', val=0)
    f2 = AutoDiff(name='y', val=np.pi/2)
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector([0, np.pi / 2])
    np.testing.assert_array_almost_equal(ad.cos(u).val, [1, 0]), 'Cosine failed'
    J, order = (ad.cos(u)).get_jacobian()
    np.testing.assert_array_equal(J, [[0, 0], [0, -1]]), 'Cosine failed'
    np.testing.assert_array_almost_equal(ad.cos(v), [1, 0]), 'Sine failed'
    J, order = (ad.cos(v)).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[0], [0]]), 'Sine failed'


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


def test_composite():
    f1 = AutoDiff(name='x', val=np.pi/4)
    f2 = AutoDiff(name='y', val=np.pi / 2)
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector([f1, np.pi])
    z = AutoDiffVector((f2, -f1))
    np.testing.assert_array_almost_equal(ad.cos(ad.sin(u)),
                                         [0.7602445970756302, 0.5403023058681398]), "Composite failed"
    J, order = (ad.cos(ad.sin(u))).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[-0.4593626849327842, 0], [0, 0]]), "Composite failed"
    np.testing.assert_array_almost_equal(ad.cos(ad.sin(v)),
                                         [0.7602445970756302, 1]), "Composite failed"
    J, order = (ad.cos(ad.sin(v))).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[-0.4593626849327842], [0]]), "Composite failed"
    np.testing.assert_array_almost_equal(u*ad.cos(ad.sin(u)),
                                         [0.597094710276033, 0.8487048774164866]), "Composite failed"
    J, order = (u*ad.cos(ad.sin(u))).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[0.3994619879961, 0], [0, 0.5403023058681397]]), "Composite failed"
    np.testing.assert_array_almost_equal(z*ad.cos(ad.sin(u)),
                                         [1.194189420552066, -0.4243524387082433]), "Composite failed"
    J, order = (z*ad.cos(ad.sin(u))).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[-0.7215652181590587, 0.7602445970756302],
                                             [-0.5403023058681398, 0]]), "Composite failed"
    np.testing.assert_array_almost_equal((z*ad.cos(ad.sin(u)))**2,
                                         [1.4260883721584792, 0.18007499223763337]), "Composite failed"
    J, order = ((z*ad.cos(ad.sin(u)))**2).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[-1.7233710995277831, 1.815752109719],
                                             [0.4585572, 0]]), "Composite failed"


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
