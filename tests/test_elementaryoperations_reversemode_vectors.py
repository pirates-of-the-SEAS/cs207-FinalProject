from ARRRtomatic_diff import AutoDiffRev, AutoDiffRevVector
import numpy as np
import math


def test_vec_add():
    f1 = AutoDiffRev(name='x', val=-1)
    f2 = AutoDiffRev(name='y', val=3)
    u = AutoDiffRevVector((f1, f2))
    v = AutoDiffRevVector((-f2, f1))
    w = AutoDiffRevVector([10, 10])
    r = AutoDiffRevVector([-3, -3])
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
    np.testing.assert_array_equal((w + r), [7, 7]), "Addition failed"
    J, order = (w + r).get_jacobian()
    np.testing.assert_array_equal(J, [[0], [0]]), "Addition failed"


def test_vec_subtract():
    f1 = AutoDiffRev(name='x', val=-1)
    f2 = AutoDiffRev(name='y', val=3)
    u = AutoDiffRevVector((f1, f2))
    v = AutoDiffRevVector((-f2, f1))
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
    f1 = AutoDiffRev(name='x', val=-1)
    f2 = AutoDiffRev(name='y', val=3)
    u = AutoDiffRevVector((f1, f2))
    v = AutoDiffRevVector((-f2, f1))
    c = AutoDiffRevVector((f1, f1, 9, 3))
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
    np.testing.assert_array_equal((c * 2), [-2, -2, 18, 6]), "Multiplication failed"


def test_vec_divide():
    f1 = AutoDiffRev(name='x', val=-1)
    f2 = AutoDiffRev(name='y', val=3)
    u = AutoDiffRevVector((f1, f2))
    v = AutoDiffRevVector((-f2, f1))
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


def test_exponentiation():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=5)
    u = AutoDiffRevVector((f1, f2))
    np.testing.assert_array_almost_equal(u ** 2, [9, 25]), "Exponentiation failed"
    J, order = (u ** 2).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[6, 0], [0, 10]]), "Exponentiation failed"
    J, order = (2 ** u).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[5.545177, 0], [0, 22.18071]]), "Exponentiation failed"


def test_abs():
    f1 = AutoDiffRev(name='x', val=-3)
    f2 = AutoDiffRev(name='y', val=5)
    r = AutoDiffRev(name='b0', val="string")
    u = AutoDiffRevVector((f1, f2))
    v = AutoDiffRevVector((r, r))
    np.testing.assert_array_equal(abs(u), [3, 5]), "Abs val failed"
    try:
        (abs(v))
    except ValueError:
        print("Caught error as expected")
    try:
        (abs(u)).get_jacobian()
    except AttributeError:
        print("Caught error as expected")


def test_eq():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=5)
    u = AutoDiffRevVector((f1, f2))
    np.testing.assert_array_equal(u, [3, 5]), "Equals failed"
    np.testing.assert_array_equal(u, u), "Equals failed"


def test_gt():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=5)
    u = AutoDiffRevVector((f1, f2))
    y = AutoDiffRevVector((f2, 8))
    assert u > [0, 0], "Greater than failed"
    assert [100, 100] > u, "Greater than failed"
    assert y > u, "Greater than failed"


def test_ge():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=5)
    u = AutoDiffRevVector((f1, f2))
    y = AutoDiffRevVector((f2, 8))
    assert u >= [0, 0], "Greater than or equal to failed"
    assert u >= [3, 5], "Greater than or equal to failed"
    assert [100, 100] >= u, "Greater than or equal to failed"
    assert y >= u, "Greater than or equal to  failed"


def test_lt():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=5)
    u = AutoDiffRevVector((f1, f2))
    y = AutoDiffRevVector((f2, 8))
    assert [0, 0] < u, "Less than failed"
    assert u < [100, 100], "Less than failed"
    assert u < y, "Less than failed"


def test_le():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=5)
    u = AutoDiffRevVector((f1, f2))
    y = AutoDiffRevVector((f2, 8))
    assert [3, 5] <= u, "Less than failed"
    assert u <= [100, 100], "Less than failed"
    assert u <= y, "Less than failed"


def test_ne():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=5)
    u = AutoDiffRevVector((f1, f2))
    y = AutoDiffRevVector((f2, 8))
    q = AutoDiffRev(name='b0', val="string")
    assert u != 11, "Not equal failed"
    assert 11 != u, "Not equal failed"
    assert u != y, "Not equal failed"
    assert y != q, "Not equal failed"


def test_modulo():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=5)
    u = AutoDiffRevVector((f1, f2))
    np.testing.assert_array_equal(u % 2, [1, 1]), "Modulo failed"
    np.testing.assert_array_equal(u % u, [0, 0]), "Modulo failed"


def test_or():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=5)
    u = AutoDiffRevVector((f1, f2))
    y = AutoDiffRevVector((f2, 8))
    assert u == [3, 5] or y == [0,0], "Or failed"
    assert u == [3, 5] or y == [5,8], "Or failed"


def test_xor():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=5)
    u = AutoDiffRevVector((f1, f2))
    y = AutoDiffRevVector((f2, 8))
    np.testing.assert_array_equal(u ^ y, [6, 13]), "Xor failed"
    np.testing.assert_array_equal(y ^ u, [6, 13]), "Xor failed"
    try:
        assert (u ^ y) == 1, "Xor failed"
    except ValueError:
        print("Caught error as expected")


def test_and():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=5)
    u = AutoDiffRevVector((f1, f2))
    y = AutoDiffRevVector((f2, 8))
    np.testing.assert_array_equal(u, u) and np.testing.assert_array_equal(y, y), "And failed"


def test_round():
    f1 = AutoDiffRev(name='x', val=3.3)
    f2 = AutoDiffRev(name='y', val=-5.8)
    u = AutoDiffRevVector((f1, f2))
    np.testing.assert_array_equal(round(u), [3, -6]), "Round failed"
    np.testing.assert_array_equal([3, -6], round(u)), "Round failed"


def test_ceil():
    f1 = AutoDiffRev(name='x', val=3.3)
    f2 = AutoDiffRev(name='y', val=-5.8)
    u = AutoDiffRevVector((f1, f2))
    np.testing.assert_array_equal(math.ceil(u), [4, -5]), "Ceil failed"


def test_floor():
    f1 = AutoDiffRev(name='x', val=3.3)
    f2 = AutoDiffRev(name='y', val=-5.8)
    u = AutoDiffRevVector((f1, f2))
    np.testing.assert_array_equal(math.floor(u), [3, -6]), "Floor failed"


def test_trunc():
    f1 = AutoDiffRev(name='x', val=3.3333)
    f2 = AutoDiffRev(name='y', val=-5.888)
    u = AutoDiffRevVector((f1, f2))
    np.testing.assert_array_equal(math.trunc(u), [3, -5]), "Floor failed"


def test_bool():
    f1 = AutoDiffRev(name='x', val=0)
    f2 = AutoDiffRev(name='y', val=0)
    u = AutoDiffRevVector((f1, f2))
    try:
        bool(u)
    except TypeError:
        print("Caught error as expected")


def test_float():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=4)
    u = AutoDiffRevVector((f1, f2))
    for val in u:
        assert type(float(val)) is float, "Float failed"


def test_named_variables():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=4)
    u = AutoDiffRevVector((f1, f2))
    assert u.get_named_variables() == {'y', 'x'}, "Named variables failed"
    assert u.variables == {'y', 'x'}, "Get variables property failed"


def test_get_value():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=4)
    u = AutoDiffRevVector((f1, f2))
    np.testing.assert_array_equal(u.get_values(), [3, 4]), "Get values failed"
    np.testing.assert_array_equal(u.val, [3, 4]), "Values property failed"


def test_contains():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=4)
    u = AutoDiffRevVector((f1, f2))
    try:
        assert 2 in u
    except NotImplementedError:
        print("Caught error as expected")


def test_shift():
    f1 = AutoDiffRev(name='x', val=3)
    f2 = AutoDiffRev(name='y', val=4)
    u = AutoDiffRevVector((f1, f2))
    np.testing.assert_array_equal(u >> 2, [16, 32]), "Shift failed"
    np.testing.assert_array_equal(u << 2, [12, 16]), "Shift failed"
    np.testing.assert_array_equal(3 >> u, [24, 32]), "Shift failed"
    np.testing.assert_array_equal(3 << u, [24, 48]), "Shift failed"


def test_neg():
    v = AutoDiffRevVector([2,2])
    np.testing.assert_array_equal(-v, [-2, -2]), "Neg failed"


def test_pos():
    v = AutoDiffRevVector([2, 2])
    np.testing.assert_array_equal(v, [2, 2]), "Pos failed"


def test_invert():
    v = AutoDiffRevVector([2, 2])
    np.testing.assert_array_equal(~v, [-3, -3]), "Invert failed"


def test_complex():
    v = AutoDiffRevVector([2, 2])
    try:
        np.testing.assert_array_equal(complex(v), [-2, -2]), "Complex failed"
    except TypeError:
        print("Caught error as expected")


def test_floordiv():
    v = AutoDiffRevVector([13, 13])
    np.testing.assert_array_equal(v//3, [4, 4]), "Neg failed"


