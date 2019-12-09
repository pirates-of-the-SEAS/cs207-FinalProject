from ARRRtomatic_diff import AutoDiffVector, AutoDiff
from ARRRtomatic_diff import functions as ad
import numpy as np
import math


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
    print(3%2)
    # np.testing.assert_array_equal(u % 2, [1, 1]), "Modulo failed"
    np.testing.assert_array_equal(u % u, [0, 0]), "Modulo failed"


def test_or():
    x = AutoDiff(name='x', val=10)
    y = AutoDiff(name='y', val=100)
    z = AutoDiff(name='z', val=10)
    assert x == 10 or y == 14, "Or failed"
    assert x == 14 or y == 100, "Or failed"
    assert x == 10 or y == 100, "Or failed"
    assert z == x or y == z, "Or failed"
    try:
        assert x == 12 or y == 12
    except AssertionError:
        print("Caught error as expected")


def test_xor():
    x = AutoDiff(name='x', val=0)
    y = AutoDiff(name='y', val=1)
    assert (x ^ y) == 1, "Xor failed"
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


def test_ceil():
    x = AutoDiff(name='x', val=10.9)
    assert 11 == math.floor(x), "Floor failed"
    assert math.floor(x) == 11, "Floor failed"


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
