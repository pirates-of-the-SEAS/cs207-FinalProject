# from ARRRtomatic_diff import AutoDiffRev, AutoDiffRevVector
# import numpy as np
#
#
# def test_instantiation_pos():
#     f1 = AutoDiffRev(name='x', val=1)
#     f2 = AutoDiffRev(name='y', val=3)
#     u = AutoDiffRevVector((f1, f2))
#     v = AutoDiffRevVector([2, 2])
#     z = AutoDiffRevVector((f1, 9))
#     q = AutoDiffRevVector((f1, f1, 9, 3))
#     np.testing.assert_array_equal(u.val, [1, 3]), "Positive instantiation failed"
#     J, order = u.get_jacobian()
#     np.testing.assert_array_equal(J, [[1, 0], [0, 1]]), "Positive instantiation failed"
#     np.testing.assert_almost_equal(v.val, [2, 2]), "Positive instantiation failed"
#     J, order = v.get_jacobian()
#     np.testing.assert_array_equal(J, [[0], [0]]), "Positive instantiation failed"
#     np.testing.assert_array_equal(z.val, [1, 9]), "Positive instantiation failed"
#     np.testing.assert_array_equal(q.val, [1, 1, 9, 3]), "Positive instantiation failed"
#
#
# def test_instantiation_neg():
#     f1 = AutoDiffRev(name='x', val=-1)
#     f2 = AutoDiffRev(name='y', val=-3)
#     u = AutoDiffRevVector((f1, f2))
#     v = AutoDiffRevVector([-2, -2])
#     np.testing.assert_array_equal(u.val, [-1,-3]), "Negative instantiation failed"
#     J, order = u.get_jacobian()
#     np.testing.assert_array_equal(J, [[1, 0], [0, 1]]), "Negative instantiation failed"
#     np.testing.assert_almost_equal(v.val, [-2, -2]), "Negative instantiation failed"
#     J, order = v.get_jacobian()
#     np.testing.assert_array_equal(J, [[0], [0]]), "Negative instantiation failed"
#
#
# def test_instantiation_zero():
#     f1 = AutoDiffRev(name='x', val=0)
#     f2 = AutoDiffRev(name='y', val=0)
#     u = AutoDiffRevVector((f1, f2))
#     np.testing.assert_array_equal(u.val, [0, 0]), "Positive instantiation failed"
#     J, order = u.get_jacobian()
#     np.testing.assert_array_equal(J, [[1, 0], [0, 1]]), "Positive instantiation failed"
#
#
# def test_instantiation_noname():
#     try:
#         AutoDiffRevVector(val=4)
#     except TypeError:
#         print("Caught error as expected")
#
#
# def test_bogus_instantiation():
#     try:
#         AutoDiffRevVector("gobbledgook")
#     except TypeError:
#         print("Caught error as expected")
#
#
# def test_empty_instantiation():
#     try:
#         AutoDiffRevVector()
#     except TypeError:
#         print("Caught error as expected")
#
#
# def test_double_instantiation():
#     try:
#         AutoDiffRevVector(name='x', val=3, trace=3)
#     except TypeError:
#         print("Caught error as expected")
#     f1 = AutoDiffRev(name='x', val=1)
#     f2 = AutoDiffRev(name='y', val=3)
#     try:
#         AutoDiffRevVector((f1, f2), f1)
#     except TypeError:
#         print("Caught error as expected")