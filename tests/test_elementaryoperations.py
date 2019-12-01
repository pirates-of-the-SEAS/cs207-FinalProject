from ARRRtomatic_diff import AutoDiff
import math
import numpy as np


def test_add():
    x = AutoDiff(name='x', val=2)
    y = AutoDiff(name='y', val=-5)
    z = AutoDiff(name='z', val=0)
    q = AutoDiff(name='b0', val="string")
    assert (x + 1) == 3, 'Addition failed'
    assert (x + 1).trace['d_x'] == 1, 'Addition failed'
    assert (1 + x) == 3, 'Addition failed'
    assert (1 + x).trace['d_x'] == 1, 'Addition failed'
    assert (y + x) == -3, 'Addition failed'
    assert (y + x).trace['d_x'] == 1, 'Addition failed'
    assert (y + x).trace['d_y'] == 1, 'Addition failed'
    assert (z + z) == 0, 'Addition failed'
    assert (z + z).trace['d_z'] == 2, 'Addition failed'
    try:
        (q + 5)
    except TypeError:
        print("Caught error as expected")


def test_subtract():
    x = AutoDiff(name='x', val=9)
    y = AutoDiff(name='y', val=-5)
    z = AutoDiff(name='z', val=0)
    q = AutoDiff(name='b0', val="string")
    assert (x - 1) == 8, 'Subtraction failed'
    assert (x - 1).trace['d_x'] == 1, 'Subtraction failed'
    assert (1 - x) == -8, 'Subtraction failed'
    assert (1 - x).trace['d_x'] == -1, 'Subtraction failed'
    assert (y - x) == -14, 'Subtraction failed'
    assert (y - x).trace['d_x'] == -1, 'Subtraction failed'
    assert (y - x).trace['d_y'] == 1, 'Subtraction failed'
    assert (z + z) == 0, 'Subtraction failed'
    assert (z + z).trace['d_z'] == 2, 'Subtraction failed'
    try:
        (q - 5)
    except TypeError:
        print("Caught error as expected")


def test_multiply():
    x = AutoDiff(name='x', val=6)
    y = AutoDiff(name='y', val=-5)
    z = AutoDiff(name='z', val=0)
    q = AutoDiff(name='b0', val="string")
    assert (x * 2) == 12, 'Multiplication failed'
    assert (x * 2) == 12, 'Multiplication failed'
    assert (x * 2).trace['d_x'] == 2, 'Multiplication failed'
    assert (2 * x) == 12, 'Multiplication failed'
    assert (2 * x).trace['d_x'] == 2, 'Multiplication failed'
    assert (y * x) == -30, 'Multiplication failed'
    assert (y * x).trace['d_x'] == -5, 'Multiplication failed'
    assert (y * x).trace['d_y'] == 6, 'Multiplication failed'
    assert (x * y) == -30, 'Multiplication failed'
    assert (x * y).trace['d_x'] == -5, 'Multiplication failed'
    assert (x * y).trace['d_y'] == 6, 'Multiplication failed'
    assert (z * z) == 0, 'Multiplication failed'
    assert (z * z).trace['d_z'] == 0, 'Multiplication failed'
    try:
        (q * 5)
    except TypeError:
        print("Caught error as expected")


def test_divide():
    x = AutoDiff(name='x', val=6)
    y = AutoDiff(name='y', val=-12)
    z = AutoDiff(name='z', val=0)
    q = AutoDiff(name='b0', val="string")
    assert (x / 2) == 3, 'Division failed'
    assert (x / 2).trace['d_x'] == (1 / 2), 'Division failed'
    assert (18 / x) == 3, 'Division failed'
    assert (18 / x).trace['d_x'] == -(1 / 2), 'Division failed'
    assert (y / x) == -2, 'Division failed'
    assert (y / x).trace['d_x'] == (12 / 36), 'Division failed'
    assert (y / x).trace['d_y'] == (1 / 6), 'Division failed'
    assert (x / y) == -0.5, 'Division failed'
    assert (x / y).trace['d_x'] == (1 / -12), 'Division failed'
    assert (x / y).trace['d_y'] == (-6 / 144), 'Division failed'
    try:
        assert (z / z) == 0
    except ZeroDivisionError as e:
        print("Caught Zero Division Error")
    try:
        assert (z / z).trace['d_z'] == 0
    except ZeroDivisionError as e:
        print("Caught Zero Division Error")
    try:
        (q / 5)
    except TypeError:
        print("Caught error as expected")


def test_exponentiation():
    x = AutoDiff(name='x', val=3)
    y = AutoDiff(name='y', val=0)
    z = AutoDiff(name='z', val=-2)
    q = AutoDiff(name='b0', val="string")
    r = AutoDiff(name='r', val=5)
    assert (x ** 2) == 9, "Exponentiation failed"
    assert (x ** 2).trace['d_x'] == 6, "Exponentiation failed"
    assert (2 ** x) == 8, "Exponentiation failed"
    assert np.allclose((2 ** x).trace['d_x'], 5.545177444479562, atol=1e-12) is True, "Exponentiation failed"
    assert (x ** 0) == 1, "Exponentiation failed"
    assert (x ** 0).trace['d_x'] == 0, "Exponentiation failed"
    assert (x ** -2) == (1 / 9), "Exponentiation failed"
    assert (x ** -2).trace['d_x'] == -2 / (3 ** 3), "Exponentiation failed"
    assert (z ** 2) == 4, "Exponentiation failed"
    assert (z ** 2).trace['d_z'] == -4, "Exponentiation failed"
    assert (z ** 3) == -8, "Exponentiation failed"
    assert (z ** 3).trace['d_z'] == 12, "Exponentiation failed"
    assert (y ** 2) == 0, "Exponentiation failed"
    assert (y ** 2).trace['d_y'] == 0, "Exponentiation failed"
    assert (x ** x) == 27, "Exponentiation failed"
    assert (r ** x) == 125, "Exponentiation failed"
    try:
        (q ** 5)
    except TypeError:
        print("Caught error as expected")


def test_abs():
    x = AutoDiff(name='x', val=-3)
    y = AutoDiff(name='y', val=3)
    z = AutoDiff(name='z', val=0)
    q = AutoDiff(name='q', val=-30.23)
    r = AutoDiff(name='b0', val="string")
    assert abs(x) == 3, "Abs val failed"
    assert abs(y) == 3, "Abs val failed"
    assert abs(z) == 0, 'Abs val failed'
    assert abs(q) == 30.23, "Abs val failed"
    try:
        (abs(r))
    except TypeError:
        print("Caught error as expected")


def test_eq():
    x = AutoDiff(name='x', val=2)
    y = AutoDiff(name='x', val=2)
    assert 2 == x, "Equals failed"
    assert x == 2, "Equals failed"
    assert x == y, "Equals failed"
    assert y == x, "Equals failed"


def test_gt():
    x = AutoDiff(name='x', val=10)
    y = AutoDiff(name='y', val=100)
    q = AutoDiff(name='b0', val="string")
    assert x > 2, "Greater than failed"
    assert 20 > x, "Greater than failed"
    assert y > x, "Greater than failed"
    try:
        (12 > q)
    except TypeError:
        print("Caught error as expected")


def test_ge():
    x = AutoDiff(name='x', val=10)
    y = AutoDiff(name='y', val=100)
    q = AutoDiff(name='b0', val="string")
    assert x >= 2, "Greater than or equal to failed"
    assert x >= 10, "Greater than or equal to failed"
    assert 20 >= x, "Greater than or equal to failed"
    assert 10 >= x, "Greater than or equal to failed"
    assert y >= x, "Greater than or equal to failed"
    try:
        (12 >= q)
    except TypeError:
        print("Caught error as expected")


def test_lt():
    x = AutoDiff(name='x', val=10)
    y = AutoDiff(name='y', val=100)
    q = AutoDiff(name='b0', val="string")
    assert 2 < x, "Less than failed"
    assert x < 20, "Less than failed"
    assert x < y, "Less than failed"
    try:
        (12 < q)
    except TypeError:
        print("Caught error as expected")


def test_le():
    x = AutoDiff(name='x', val=10)
    y = AutoDiff(name='y', val=100)
    q = AutoDiff(name='b0', val="string")
    assert 2 <= x, "Less than or equal to failed"
    assert 10 <= x, "Less than or equal to failed"
    assert x <= 20, "Less than or equal to failed"
    assert x <= 10, 'Less than or equal to failed'
    assert x <= y, "Less than or equal to failed"
    try:
        (12 <= q)
    except TypeError:
        print("Caught error as expected")


def test_ne():
    x = AutoDiff(name='x', val=10)
    y = AutoDiff(name='y', val=100)
    q = AutoDiff(name='b0', val="string")
    assert x != 11, "Not equal failed"
    assert 11 != x, "Not equal failed"
    assert x != y, "Not equal failed"
    assert 12 != q, "Not equal failed"


def test_modulo():
    x = AutoDiff(name='x', val=15)
    y = AutoDiff(name='y', val=12)
    z = AutoDiff(name='z', val=4)
    q = AutoDiff(name='b0', val="string")
    assert x % 2 == 1, "Modulo failed"
    assert y % 2 == 0, "Modulo failed"
    assert y % z == 0, "Modulo failed"
    assert x % z == 3, "Modulo failed"
    try:
        (12 % q)
    except TypeError:
        print("Caught error as expected")
    try:
        (q % 4)
    except TypeError:
        print("Caught error as expected")


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
