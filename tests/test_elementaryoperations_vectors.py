from ARRRtomatic_diff import AutoDiff, AutoDiffVector
import numpy as np
import math


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


def test_exponentiation():
    f1 = AutoDiff(name='x', val=3)
    f2 = AutoDiff(name='y', val=5)
    u = AutoDiffVector((f1, f2))
    np.testing.assert_array_almost_equal(u ** 2, [9, 25]), "Exponentiation failed"
    J, order = (u ** 2).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[6, 0], [0, 10]]), "Exponentiation failed"
    J, order = (2 ** u).get_jacobian()
    np.testing.assert_array_almost_equal(J, [[5.545177, 0], [0, 22.18071]]), "Exponentiation failed"


def test_abs():
    f1 = AutoDiff(name='x', val=-3)
    f2 = AutoDiff(name='y', val=5)
    r = AutoDiff(name='b0', val="string")
    u = AutoDiffVector((f1, f2))
    v = AutoDiffVector((r, r))
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
    f1 = AutoDiff(name='x', val=3)
    f2 = AutoDiff(name='y', val=5)
    u = AutoDiffVector((f1, f2))
    np.testing.assert_array_equal(u, [3, 5]), "Equals failed"
    np.testing.assert_array_equal(u, u), "Equals failed"


def test_gt():
    f1 = AutoDiff(name='x', val=3)
    f2 = AutoDiff(name='y', val=5)
    u = AutoDiffVector((f1, f2))
    y = AutoDiffVector((f2, 8))
    assert u > [0, 0], "Greater than failed"
    assert [100, 100] > u, "Greater than failed"
    assert y > u, "Greater than failed"


def test_ge():
    f1 = AutoDiff(name='x', val=3)
    f2 = AutoDiff(name='y', val=5)
    u = AutoDiffVector((f1, f2))
    y = AutoDiffVector((f2, 8))
    assert u >= [0, 0], "Greater than or equal to failed"
    assert u >= [3, 5], "Greater than or equal to failed"
    assert [100, 100] >= u, "Greater than or equal to failed"
    assert y >= u, "Greater than or equal to  failed"


def test_lt():
    f1 = AutoDiff(name='x', val=3)
    f2 = AutoDiff(name='y', val=5)
    u = AutoDiffVector((f1, f2))
    y = AutoDiffVector((f2, 8))
    assert [0, 0] < u, "Less than failed"
    assert u < [100, 100], "Less than failed"
    assert u < y, "Less than failed"


def test_le():
    f1 = AutoDiff(name='x', val=3)
    f2 = AutoDiff(name='y', val=5)
    u = AutoDiffVector((f1, f2))
    y = AutoDiffVector((f2, 8))
    assert [3, 5] <= u, "Less than failed"
    assert u <= [100, 100], "Less than failed"
    assert u <= y, "Less than failed"


def test_ne():
    f1 = AutoDiff(name='x', val=3)
    f2 = AutoDiff(name='y', val=5)
    u = AutoDiffVector((f1, f2))
    y = AutoDiffVector((f2, 8))
    q = AutoDiff(name='b0', val="string")
    assert u != 11, "Not equal failed"
    assert 11 != u, "Not equal failed"
    assert u != y, "Not equal failed"
    assert y != q, "Not equal failed"


def test_modulo():
    f1 = AutoDiff(name='x', val=3)
    f2 = AutoDiff(name='y', val=5)
    u = AutoDiffVector((f1, f2))
    np.testing.assert_array_equal(u % 2, [1, 1]), "Modulo failed"
    np.testing.assert_array_equal(u % u, [0, 0]), "Modulo failed"


def test_or():
    f1 = AutoDiff(name='x', val=3)
    f2 = AutoDiff(name='y', val=5)
    u = AutoDiffVector((f1, f2))
    y = AutoDiffVector((f2, 8))
    assert u == [3, 5] or y == [0,0], "Or failed"
    assert u == [3, 5] or y == [5,8], "Or failed"


def test_xor():
    f1 = AutoDiff(name='x', val=3)
    f2 = AutoDiff(name='y', val=5)
    u = AutoDiffVector((f1, f2))
    y = AutoDiffVector((f2, 8))
    assert (u ^ y) == 1, "Xor failed"
    assert (x ^ 1) == 1, "Xor failed"
    assert (1 ^ x) == 1, "Xor failed"
    try:
        assert x ^ x == 1
    except AssertionError:
        print("Caught Error as expected")


def test_and():
    x = AutoDiff(name='x', val=10)
    y = AutoDiff(name='y', val=100)
    z = AutoDiff(name='z', val=10)
    q = AutoDiff(name='q', val=10)
    assert x and 10 and y == 100, "Or failed"
    assert z == x and q == z, "Or failed"
    try:
        assert x == 12 and y == 12
    except AssertionError:
        print("Caught error as expected")
    try:
        assert x == 10 and y == 12
    except AssertionError:
        print("Caught error as expected")
    try:
        assert x == 12 and y == 100
    except AssertionError:
        print("Caught error as expected")


def test_round():
    x = AutoDiff(name='x', val=5.7)
    y = AutoDiff(name='y', val=-4.6)
    z = AutoDiff(name='z', val=0)
    assert round(x) == 6, "Round failed"
    assert -5 == round(y), "Round failed"
    assert 0 == round(z), "Round failed"


def test_ceil():
    x = AutoDiff(name='x', val=10.1)
    assert 11 == math.ceil(x), "Ceil failed"
    assert math.ceil(x) == 11, "Ceil failed"


def test_floor():
    x = AutoDiff(name='x', val=10.9)
    assert 10 == math.floor(x), "Floor failed"
    assert math.floor(x) == 10, "Floor failed"


def test_trunc():
    x = AutoDiff(name='x', val=-4.343)
    assert -4 == math.trunc(x), "Truncate failed"
    assert math.trunc(x) == -4, "Truncate failed"


def test_str():
    x = AutoDiff(name='x', val=2)
    assert str(x) == "{'val': 2, 'd_x': 1}", "Str failed"


def test_bool():
    x = AutoDiff(name='x', val=13)
    y = AutoDiff(name='x', val=0)
    assert bool(x) == True, "Bool failed"
    assert bool(y) == False, "Bool failed"


def test_float():
    x = AutoDiff(name='x', val=3)
    assert type(float(x)) is float, "Float failed"


def test_named_variables():
    x = AutoDiff(name='x', val=3)
    assert x.get_named_variables() == {'x'}, "Named variables failed"
    assert x.variables == {'x'}, "Get variables property failed"


def test_get_trace():
    x = AutoDiff(name='x', val=3)
    assert x.get_trace() == {'d_x': 1, 'val': 3}, "Get trace failed"


def test_get_value():
    x = AutoDiff(name='x', val=3)
    assert x.get_value() == 3, "Get value failed"
    assert x.val == 3, "Get value property failed"


def test_get_gradient():
    x = AutoDiff(name='x', val=3)
    grad1, varnames = x.get_gradient()
    grad2, _ = (8 * x).get_gradient()
    grad3, _ = x.gradient

    assert np.allclose(grad1, np.array([1.])), "Get gradient failed"
    assert np.allclose(grad2, np.array([8.])), "Get gradient failed"
    assert np.allclose(grad3, np.array([1.])), "Get gradient property failed"


def test_contains():
    x = AutoDiff(name='x', val=2)
    assert x in [2], "Contains failed"


def test_shift():
    x = AutoDiff(name='x', val=5)
    assert x >> 2 == 1, "Shift failed"
    assert x << 10 == 5120, "Shift failed"
    assert 2 >> x == 0, "Shift failed"
    assert 2 << x == 64, "Shift failed"


# def test_getitem():
#     x = AutoDiff(name='x', val=13)
#     assert x['val'] == 13, "Get item failed"
#     assert x['d_x'] == 1, "Get item failed"


# def test_setitem():
#     x = AutoDiff(name='x', val=13)
#     x['val'] = 2
#     assert x['val'] == 2, "Set item failed"
#     x['d_x'] = 24.7
#     assert x['d_x'] == 24.7, "Set item failed"


def test_repr():
    x = AutoDiff(name='x', val=13)
    assert repr(x) == """AutoDiff(names_init_vals={\'x\': 13}, trace="{\'val\': 13, \'d_x\': 1}")""", "Repr failed"


def test_neg():
    x = AutoDiff(name='x', val=2)
    assert -x == -2, "Neg failed"
    assert -x.trace['d_x'] == -1, "Neg failed"


def test_pos():
    x = AutoDiff(name='x', val=2)
    assert x == 2, "Pos failed"


def test_invert():
    x = AutoDiff(name='x', val=2)
    assert ~x == -3, "Invert failed"


def test_complex():
    x = AutoDiff(name='x', val=2)
    assert complex(x) == (2+0j), "Complex failed"


def test_floordiv():
    x = AutoDiff(name='x', val=13)
    assert x // 3 == 4, 'Floordiv failed'
    assert 160 // x == 12, "Floordiv failed"


