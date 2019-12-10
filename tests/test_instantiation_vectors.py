from ARRRtomatic_diff import AutoDiff, AutoDiffVector
import numpy as np


def test_instantiation_pos():
    f1 = AutoDiff(name='x', val=1)
    f2 = AutoDiff(name='y', val=3)
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector([2, 2])
    z = AutoDiffVector((f1, 9))
    q = AutoDiffVector((f1, f1, 9, 3))
    np.testing.assert_array_equal(u.val, [1,3]), "Positive instantiation failed"
    J, order = u.get_jacobian()
    np.testing.assert_array_equal(J, [[1, 0], [0, 1]]), "Positive instantiation failed"
    np.testing.assert_almost_equal(v.val, [2, 2]), "Positive instantiation failed"
    J, order = v.get_jacobian()
    np.testing.assert_array_equal(J, [[0], [0]]), "Positive instantiation failed"
    np.testing.assert_array_equal(z.val, [1, 9]), "Positive instantiation failed"
    np.testing.assert_array_equal(q.val, [1, 1, 9, 3]), "Positive instantiation failed"


def test_instantiation_neg():
    f1 = AutoDiff(name='x', val=-1)
    f2 = AutoDiff(name='y', val=-3)
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector([-2, -2])
    np.testing.assert_array_equal(u.val, [-1,-3]), "Negative instantiation failed"
    J, order = u.get_jacobian()
    np.testing.assert_array_equal(J, [[1, 0], [0, 1]]), "Negative instantiation failed"
    np.testing.assert_almost_equal(v.val, [-2, -2]), "Negative instantiation failed"
    J, order = v.get_jacobian()
    np.testing.assert_array_equal(J, [[0], [0]]), "Negative instantiation failed"


def test_instantiation_zero():
    f1 = AutoDiff(name='x', val=0)
    f2 = AutoDiff(name='y', val=0)
    u = AutoDiffVector((f1, f2))
    np.testing.assert_array_equal(u.val, [0, 0]), "Positive instantiation failed"
    J, order = u.get_jacobian()
    np.testing.assert_array_equal(J, [[1, 0], [0, 1]]), "Positive instantiation failed"


def test_instantiation_noname():
    try:
        AutoDiffVector(val=4)
    except TypeError:
        print("Caught error as expected")


def test_bogus_instantiation():
    try:
        AutoDiffVector(auto_diff_variables=9)
    except TypeError:
        print("Caught error as expected")


def test_empty_instantiation():
    try:
        AutoDiffVector()
    except TypeError:
        print("Caught error as expected")


def test_double_instantiation():
    try:
        AutoDiffVector(name='x', val=3, trace=3)
    except TypeError:
        print("Caught error as expected")
    f1 = AutoDiff(name='x', val=1)
    f2 = AutoDiff(name='y', val=3)
    try:
        AutoDiffVector((f1, f2), f1)
    except TypeError:
        print("Caught error as expected")


def test_duplicate_instantiation():
    f1 = AutoDiff(name='x', val=1)
    f2 = AutoDiff(name='x', val=3)
    try:
        AutoDiffVector((f1, f2))
    except Exception:
        print("Caught error as expected")
