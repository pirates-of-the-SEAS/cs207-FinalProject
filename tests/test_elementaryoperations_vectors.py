from ARRRtomatic_diff import AutoDiff, AutoDiffVector
from ARRRtomatic_diff.functions import sin, exp, sqrt
import numpy as np


def test_vec_add():
    x = AutoDiff(name='x', val=-1)
    y = AutoDiff(name='y', val=3)
    u = AutoDiffVector((x, y))
    v = AutoDiffVector((-y, x))
    q = [2, 1.5]
    np.testing.assert_array_equal((u + q).val, [1, 4.5]), 'Addition failed'
    np.testing.assert_array_equal((q + u).val, [1, 4.5]), 'Addition failed'
    J, order = (u + q).get_jacobian()
    np.testing.assert_array_equal(J, [[1, 0],[0,1]]), 'Addition failed'
    J, order = (q + u).get_jacobian()
    np.testing.assert_array_equal(J, [[1, 0], [0, 1]]), 'Addition failed'
    np.testing.assert_array_equal((u + v).val, [-4,  2]), 'Addition failed'
    np.testing.assert_array_equal((v + u).val, [-4, 2]), 'Addition failed'
    J, order = (u + v).get_jacobian()
    np.testing.assert_array_equal(J, [[1, -1],[1, 1]]), 'Addition failed'







