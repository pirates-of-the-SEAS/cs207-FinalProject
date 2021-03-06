from ARRRtomatic_diff import AutoDiffRev
import math
import numpy as np


def test_add():
    x = AutoDiffRev(name='x', val=2)
    y = AutoDiffRev(name='y', val=-5)
    z = AutoDiffRev(name='z', val=0)
    q = AutoDiffRev(name='b0', val="string")
    assert (x + 1).get_value() == 3, 'Addition failed'
    g, _ = (x + 1).get_gradient()
    assert g == [1], 'Addition failed'
    assert (1 + x) == 3, 'Addition failed'
    g, _ = (1 + x).get_gradient()
    assert g == [1], "Addition failed"
    assert (y + x).get_value() == -3, 'Addition failed'
    g, _ = (y + x).get_gradient()
    np.testing.assert_array_almost_equal(g, [1, 1]), 'Addition failed'
    assert (z + z).get_value() == 0, 'Addition failed'
    g, _ = (z + z).get_gradient()
    assert g == [2], 'Addition failed'
    try:
        (q + 5)
    except TypeError:
        print("Caught error as expected")


def test_subtract():
    x = AutoDiffRev(name='x', val=9)
    y = AutoDiffRev(name='y', val=-5)
    z = AutoDiffRev(name='z', val=0)
    q = AutoDiffRev(name='b0', val="string")
    assert (x - 1) == 8, 'Subtraction failed'
    g, _ = (x - 1).get_gradient()
    assert g == [1], "Subtraction failed"
    assert (1 - x) == -8, 'Subtraction failed'
    g, _ = (1 - x).get_gradient()
    assert g == [-1], "Subtraction failed"
    assert (y - x) == -14, 'Subtraction failed'
    g, _ = (y - x).get_gradient()
    np.testing.assert_array_equal(g, [-1, 1]), 'Subtraction failed'
    assert (z + z) == 0, 'Subtraction failed'
    g, _ = (z + z).get_gradient()
    assert g == 2, 'Subtraction failed'
    try:
        (q - 5)
    except TypeError:
        print("Caught error as expected")


def test_multiply():
    x = AutoDiffRev(name='x', val=6)
    y = AutoDiffRev(name='y', val=-5)
    z = AutoDiffRev(name='z', val=0)
    q = AutoDiffRev(name='b0', val="string")
    assert (x * 2) == 12, 'Multiplication failed'
    g, _ = (x * 2).get_gradient()
    assert g == [2], "Multiplication failed"
    assert (2 * x) == 12, 'Multiplication failed'
    g, _ = (2 * x).get_gradient()
    assert g == [2], "Multiplication failed"
    assert (y * x).get_value() == -30, 'Multiplication failed'
    g, _ = (y * x).get_gradient()
    np.testing.assert_array_equal(g, [-5, 6]), 'Multiplication failed'
    assert (x * y).get_value() == -30, 'Multiplication failed'
    g, _ = (x * y).get_gradient()
    np.testing.assert_array_equal(g, [-5, 6]), 'Multiplication failed'
    assert (z * z).get_value() == 0, 'Multiplication failed'
    g, _ = (z * z).get_gradient()
    np.testing.assert_array_equal(g, [0]), 'Multiplication failed'
    try:
        (q * 5).get_value()
    except TypeError:
        print("Caught error as expected")


def test_divide():
    x = AutoDiffRev(name='x', val=6)
    y = AutoDiffRev(name='y', val=-12)
    z = AutoDiffRev(name='z', val=0)
    q = AutoDiffRev(name='b0', val="string")
    assert (x / 2) == 3, 'Division failed'
    g, _ = (x / 2).get_gradient()
    assert g == [1 / 2], 'Division failed'
    assert (18 / x) == 3, 'Division failed'
    g, _ = (18 / x).get_gradient()
    assert g == [-1/2], "Division failed"
    assert (y / x) == -2, 'Division failed'
    g, _ = (y / x).get_gradient()
    np.testing.assert_array_equal(g, [(12 / 36), (1 / 6)]), 'Division failed'
    assert (x / y).get_value() == -0.5, 'Division failed'
    g, _ = (x / y).get_gradient()
    np.testing.assert_array_equal(g, [(1 / -12), (-6 / 144)]), 'Division failed'
    try:
        assert (z / z).get_value() == 0
    except ZeroDivisionError:
        print("Caught Zero Division Error")
    try:
        assert (z / z).get_gradient() == 0
    except ZeroDivisionError:
        print("Caught Zero Division Error")
    try:
        (q / 5).get_value()
    except TypeError:
        print("Caught error as expected")


def test_composition():
    x = AutoDiffRev(name='x', val=2)
    y = AutoDiffRev(name='y', val=-5)
    z = AutoDiffRev(name='z', val=0)
    assert ((y + x) * y * y).get_value() == -75, 'Composition failed'
    g, _ = ((y + x) * y * y).get_gradient()
    np.testing.assert_array_almost_equal(g, [25, 55]), "Composition failed"
    assert ((y - x) * y) == 35, 'Composition failed'
    g, _ = ((y - x) * y * y).get_gradient()
    np.testing.assert_array_almost_equal(g, [-25, 95]), "Composition failed"
    try:
        assert ((y - x) * y * y) / z == 0
    except ZeroDivisionError:
        print("Caught Zero Division Error")


def test_exponentiation():
    x = AutoDiffRev(name='x', val=3)
    y = AutoDiffRev(name='y', val=0)
    z = AutoDiffRev(name='z', val=-2)
    q = AutoDiffRev(name='b0', val="string")
    r = AutoDiffRev(name='r', val=5)
    assert (x ** 2) == 9, "Exponentiation failed"
    g, _ = (x ** 2).get_gradient()
    assert g == [6], "Exponentiation failed"
    assert (2 ** x) == 8, "Exponentiation failed"
    g, _ = (2 ** x).get_gradient()
    np.testing.assert_array_almost_equal(g, [5.545177444479562]), "Exponentiation failed"
    assert (x ** 0) == 1, "Exponentiation failed"
    g, _ = (x ** 0).get_gradient()
    np.testing.assert_array_almost_equal(g, [0]), "Exponentiation failed"
    assert (x ** -2) == (1 / 9), "Exponentiation failed"
    g, _ = (x ** -2).get_gradient()
    np.testing.assert_array_almost_equal(g, [-2 / (3 ** 3)]), "Exponentiation failed"
    assert (z ** 2) == 4, "Exponentiation failed"
    g, _ = (z ** 2).get_gradient()
    np.testing.assert_array_almost_equal(g, [-4]), "Exponentiation failed"
    assert (z ** 3) == -8, "Exponentiation failed"
    g, _ = (z ** 3).get_gradient()
    np.testing.assert_array_almost_equal(g, [12]), "Exponentiation failed"
    assert (y ** 2) == 0, "Exponentiation failed"
    g, _ = (y ** 2).get_gradient()
    np.testing.assert_array_almost_equal(g, [0]), "Exponentiation failed"
    assert (x ** x).get_value() == 27, "Exponentiation failed"
    g, _ = (x ** x).get_gradient()
    assert g == [56.66253179403897], "Exponentiation failed"
    assert (r ** x) == 125, "Exponentiation failed"
    g, _ = (r ** x).get_gradient()
    np.testing.assert_array_almost_equal(g, [75, 201.17973905426254]), "Exponentiation failed"
    try:
        (q ** 5)
    except TypeError:
        print("Caught error as expected")


def test_abs():
    x = AutoDiffRev(name='x', val=-3)
    y = AutoDiffRev(name='y', val=3)
    z = AutoDiffRev(name='z', val=0)
    q = AutoDiffRev(name='q', val=-30.23)
    r = AutoDiffRev(name='b0', val="string")
    assert abs(x) == 3, "Abs val failed"
    assert abs(y) == 3, "Abs val failed"
    assert abs(z) == 0, 'Abs val failed'
    assert abs(q) == 30.23, "Abs val failed"
    try:
        (abs(r))
    except TypeError:
        print("Caught error as expected")


def test_eq():
    x = AutoDiffRev(name='x', val=2)
    y = AutoDiffRev(name='x', val=2)
    assert 2 == x, "Equals failed"
    assert x == 2, "Equals failed"
    assert x == y, "Equals failed"
    assert y == x, "Equals failed"


def test_gt():
    x = AutoDiffRev(name='x', val=10)
    y = AutoDiffRev(name='y', val=100)
    q = AutoDiffRev(name='b0', val="string")
    assert x > 2, "Greater than failed"
    assert 20 > x, "Greater than failed"
    assert y > x, "Greater than failed"
    try:
        (12 > q)
    except TypeError:
        print("Caught error as expected")


def test_ge():
    x = AutoDiffRev(name='x', val=10)
    y = AutoDiffRev(name='y', val=100)
    q = AutoDiffRev(name='b0', val="string")
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
    x = AutoDiffRev(name='x', val=10)
    y = AutoDiffRev(name='y', val=100)
    q = AutoDiffRev(name='b0', val="string")
    assert 2 < x, "Less than failed"
    assert x < 20, "Less than failed"
    assert x < y, "Less than failed"
    try:
        (12 < q)
    except TypeError:
        print("Caught error as expected")


def test_le():
    x = AutoDiffRev(name='x', val=10)
    y = AutoDiffRev(name='y', val=100)
    q = AutoDiffRev(name='b0', val="string")
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
    x = AutoDiffRev(name='x', val=10)
    y = AutoDiffRev(name='y', val=100)
    q = AutoDiffRev(name='b0', val="string")
    assert x != 11, "Not equal failed"
    assert 11 != x, "Not equal failed"
    assert x != y, "Not equal failed"
    assert 12 != q, "Not equal failed"


def test_modulo():
    x = AutoDiffRev(name='x', val=15)
    y = AutoDiffRev(name='y', val=12)
    z = AutoDiffRev(name='z', val=4)
    q = AutoDiffRev(name='b0', val="string")
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
    x = AutoDiffRev(name='x', val=10)
    y = AutoDiffRev(name='y', val=100)
    z = AutoDiffRev(name='z', val=10)
    assert x == 10 or y == 14, "Or failed"
    assert x == 14 or y == 100, "Or failed"
    assert x == 10 or y == 100, "Or failed"
    assert z == x or y == z, "Or failed"
    try:
        assert x == 12 or y == 12
    except AssertionError:
        print("Caught error as expected")


def test_xor():
    x = AutoDiffRev(name='x', val=0)
    y = AutoDiffRev(name='y', val=1)
    assert (x ^ y) == 1, "Xor failed"
    assert (x ^ 1) == 1, "Xor failed"
    assert (1 ^ x) == 1, "Xor failed"
    try:
        assert x ^ x == 1
    except AssertionError:
        print("Caught Error as expected")


def test_and():
    x = AutoDiffRev(name='x', val=10)
    y = AutoDiffRev(name='y', val=100)
    z = AutoDiffRev(name='z', val=10)
    q = AutoDiffRev(name='q', val=10)
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
    x = AutoDiffRev(name='x', val=5.7)
    y = AutoDiffRev(name='y', val=-4.6)
    z = AutoDiffRev(name='z', val=0)
    assert round(x) == 6, "Round failed"
    assert -5 == round(y), "Round failed"
    assert 0 == round(z), "Round failed"


def test_ceil():
    x = AutoDiffRev(name='x', val=10.1)
    assert 11 == math.ceil(x), "Ceil failed"
    assert math.ceil(x) == 11, "Ceil failed"


def test_floor():
    x = AutoDiffRev(name='x', val=10.9)
    assert 10 == math.floor(x), "Floor failed"
    assert math.floor(x) == 10, "Floor failed"


def test_trunc():
    x = AutoDiffRev(name='x', val=-4.343)
    assert -4 == math.trunc(x), "Truncate failed"
    assert math.trunc(x) == -4, "Truncate failed"


def test_str():
    x = AutoDiffRev(name='x', val=2)
    assert str(x) == "2", "Str failed"


def test_bool():
    x = AutoDiffRev(name='x', val=13)
    y = AutoDiffRev(name='x', val=0)
    assert bool(x) == True, "Bool failed"
    assert bool(y) == False, "Bool failed"


def test_float():
    x = AutoDiffRev(name='x', val=3)
    assert type(float(x)) is float, "Float failed"


def test_named_variables():
    x = AutoDiffRev(name='x', val=3)
    assert x.get_named_variables() == {'x'}, "Named variables failed"
    assert x.variables == {'x'}, "Get variables property failed"


def test_get_value():
    x = AutoDiffRev(name='x', val=3)
    assert x.get_value() == 3, "Get value failed"
    assert x.val == 3, "Get value property failed"


def test_contains():
    x = AutoDiffRev(name='x', val=2)
    assert x in [2], "Contains failed"


def test_shift():
    x = AutoDiffRev(name='x', val=5)
    assert x >> 2 == 1, "Shift failed"
    assert x << 10 == 5120, "Shift failed"
    assert 2 >> x == 0, "Shift failed"
    assert 2 << x == 64, "Shift failed"


def test_neg():
    x = AutoDiffRev(name='x', val=2)
    assert -x == -2, "Neg failed"
    g, _ = (-x).get_gradient()
    assert g == [-1], "Neg failed"


def test_pos():
    x = AutoDiffRev(name='x', val=2)
    assert x == 2, "Pos failed"
    assert x.__pos__() == 2, "Pos failed"


def test_invert():
    x = AutoDiffRev(name='x', val=2)
    assert ~x == -3, "Invert failed"

def test_andor():
    x = AutoDiffRev(name='x', val=2)
    assert x.__ror__(3) == 3, "ror failed"
    assert x.__rand__(3) == 2, "rand failed"

def test_complex():
    x = AutoDiffRev(name='x', val=2)
    assert complex(x) == (2 + 0j), "Complex failed"


def test_floordiv():
    x = AutoDiffRev(name='x', val=13)
    assert x // 3 == 4, 'Floordiv failed'
    assert 160 // x == 12, "Floordiv failed"
