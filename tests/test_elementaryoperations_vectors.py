from ARRRtomatic_diff import AutoDiff, AutoDiffVector
from ARRRtomatic_diff.functions import sin, exp, sqrt
import numpy as np


def test__vec_add():
    f1 = AutoDiff(name='x', val=-1)
    f2 = AutoDiff(name='y', val=3)
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector((-f2, f1))
    q = [2, 1.5]
    np.testing.assert_array_equal((u + q).val, [1, 4.5]), 'Addition failed'
    np.testing.assert_array_equal((q + u).val, [1, 4.5]), 'Addition failed'
    J, order = (u + q).get_jacobian()
    np.testing.assert_array_equal(J, [[1, 0], [0, 1]]), 'Addition failed'
    J, order = (v + q).get_jacobian()
    np.testing.assert_array_equal(J, [[0, -1], [1, 0]]), 'Addition failed'
    J, order = (q + u).get_jacobian()
    np.testing.assert_array_equal(J, [[1, 0], [0, 1]]), 'Addition failed'
    np.testing.assert_array_equal((u + v).val, [-4,  2]), 'Addition failed'
    np.testing.assert_array_equal((v + u).val, [-4, 2]), 'Addition failed'
    J, order = (u + v).get_jacobian()
    np.testing.assert_array_equal(J, [[1, -1], [1, 1]]), 'Addition failed'

def test_vec_subtract():
    f1 = AutoDiff(name='x', val=-1)
    f2 = AutoDiff(name='y', val=3)
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector((-f2, f1))
    q = [2, 1.5]
    np.testing.assert_array_equal((u - q).val, [-3, 1.5]), 'Subtraction failed'
    np.testing.assert_array_equal((q - u).val, [3, -1.5]), 'Subtraction failed'
    J, order = (u - q).get_jacobian()
    np.testing.assert_array_equal(J, [[1, 0],[0,1]]), 'Subtraction failed'
    J, order = (q - u).get_jacobian()
    np.testing.assert_array_equal(J, [[-1, 0], [0, -1]]), 'Subtraction failed'
    np.testing.assert_array_equal((u - v).val, [2,  4]), 'Subtraction failed'
    np.testing.assert_array_equal((v - u).val, [-2, -4]), 'Subtraction failed'
    J, order = (u - v).get_jacobian()
    np.testing.assert_array_equal(J, [[1, 1], [-1, 1]]), 'Subtraction failed'


def test_vec_multiply():
    f1 = AutoDiff(name='x', val=-1)
    f2 = AutoDiff(name='y', val=3)
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector((-f2, f1))
    q = [2, 0]
    t = [4, 4]
    np.testing.assert_array_equal((u * 3).val, [-3, 9]), 'Multiplication failed'
    J, order = (u * 3).get_jacobian()
    np.testing.assert_array_equal(J, [[3, 0], [0, 3]]), "Multiplication failed"
    np.testing.assert_array_equal((-4 * u).val, [4, -12]), 'Multiplication failed'
    np.testing.assert_array_equal((u * q).val, [-2, 0]), 'Multiplication failed'
    np.testing.assert_array_equal((q * u).val, [-2, 0]), 'Multiplication failed'
    J, order = (u * t).get_jacobian()
    np.testing.assert_array_equal(J, [[4, 0], [0, 4]]), "Multiplication failed"
    J, order = (u * q).get_jacobian()
    np.testing.assert_array_equal(J, [[2, 0], [0, 0]]), 'Multiplication failed'
    J, order = (q * u).get_jacobian()
    np.testing.assert_array_equal(J, [[2, 0], [0, 0]]), 'Multiplication failed'
    J, order = (u * v).get_jacobian()
    np.testing.assert_array_equal(J, [[-3, 1], [3, -1]]), 'Multiplication failed'
    J, order = (v * u).get_jacobian()
    np.testing.assert_array_equal(J, [[-3, 1], [3, -1]]), 'Multiplication failed'


def test_vec_divide():
    f1 = AutoDiff(name='x', val=-1)
    f2 = AutoDiff(name='y', val=3)
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector((-f2, f1))
    q = [2, 1]
    t = [4, 4]
    np.testing.assert_array_equal((u / 3).val, [-(1/3), 1]), 'Multiplication failed'
    J, order = (u / 3).get_jacobian()
    np.testing.assert_array_equal(J, [[(1/3), 0], [0, (1/3)]]), "Multiplication failed"
    np.testing.assert_array_equal((-4 / u).val, [4, -(4/3)]), 'Multiplication failed'
    np.testing.assert_array_equal((u / q).val, [-0.5, 3]), 'Multiplication failed'
    np.testing.assert_array_equal((q / u).val, [-2, (1/3)]), 'Multiplication failed'
    J, order = (u / t).get_jacobian()
    np.testing.assert_array_equal(J, [[(1/4), 0], [0, (1/4)]]), "Multiplication failed"
    J, order = (u / v).get_jacobian()
    np.testing.assert_array_equal(J, [[-(1/3), -(1/9)], [-3, -1]]), 'Multiplication failed'




