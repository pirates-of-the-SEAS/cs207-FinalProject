from ARRRtomatic_diff import AutoDiff, AutoDiffVector
import numpy as np

#
# def test_instantiation_pos():
#     f1 = AutoDiff(name='x', val=1)
#     f2 = AutoDiff(name='y', val=3)
#     u = AutoDiffVector((f1, f2))
#     v = AutoDiffVector([2, 2])
#     np.testing.assert_array_equal(u.val, [1,3]), "Positive instantiation failed"
#     J, order = u.get_jacobian()
#     np.testing.assert_array_equal(J, [[1, 0], [0, 1]]), "Positive instantiation failed"
#     # np.testing.assert_almost_equal(v.val, [2, 2]), "Positive instantiation failed"
#
#
# def test_instantiation_neg():
#     f1 = AutoDiff(name='x', val=-1)
#     f2 = AutoDiff(name='y', val=-3)
#     u = AutoDiffVector((f1, f2))
#     np.testing.assert_array_equal(u.val, [-1,-3]), "Negative instantiation failed"
#     J, order = u.get_jacobian()
#     np.testing.assert_array_equal(J, [[1, 0], [0, 1]]), "Negative instantiation failed"
#
#
# def test_instantiation_zero():
#     x = AutoDiffVector(name='x', val=0)
#     assert x.trace['val'] == 0, 'Zero instantiation failed'
#     assert x.trace['d_x'] == 1, 'Zero instantiation failed'
#
# def test_instantiation_noname():
#     try:
#         AutoDiffVector(val=4)
#     except ValueError:
#         print("Caught error as expected")
#
# def test_bogus_instantiation():
#     try:
#         AutoDiffVector("gobbledgook")
#     except TypeError:
#         print("Caught error as expected")
#
#
# def test_empty_instantiation():
#     try:
#         AutoDiffVector()
#     except ValueError:
#         print("Caught error as expected")
#
#
# def test_double_instantiation():
#     try:
#         AutoDiffVector(name='x', val=3, trace=3)
#     except ValueError:
#         print("Caught error as expected")
#
# def test_nameless_instantiation():
#     try:
#         AutoDiffVector(val=3)
#     except ValueError:
#         print("Caught error as expected")