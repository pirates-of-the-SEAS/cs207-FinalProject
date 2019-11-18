from ARRRtomatic_diff import AutoDiff
import math

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
   assert (x / 2).trace['d_x'] == (1/2), 'Division failed'
   assert (18 / x) == 3, 'Division failed'
   assert (18 / x).trace['d_x'] == -(1/2), 'Division failed'
   assert (y / x) == -2, 'Division failed'
   assert (y / x).trace['d_x'] == (12/36), 'Division failed'
   assert (y / x).trace['d_y'] == (1/6), 'Division failed'
   assert (x / y) == -0.5, 'Division failed'
   assert (x / y).trace['d_x'] == (1/-12), 'Division failed'
   assert (x / y).trace['d_y'] == (-6/144), 'Division failed'
   try:
       assert (z / z) == 0
   except ZeroDivisionError as e:
       print("Caught Zero Division Error")
   try:
       assert (z / z).trace['d_z'] == 0
   except ZeroDivisionError as e:
       print("Caught Zero Division Error")
   try:
      (q * 5)
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
    assert (x ** 0) == 1, "Exponentiation failed"
    assert (x ** 0).trace['d_x'] == 0, "Exponentiation failed"
    assert (x ** -2) == (1/9), "Exponentiation failed"
    assert (x ** -2).trace['d_x'] == -2/(3**3), "Exponentiation failed"
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

def test_req():
   x = AutoDiff(name='x', val=2)
   assert 2 == x, "Reverse equals failed"

def test_gt():
   x = AutoDiff(name='x', val=10)
   y = AutoDiff(name='y', val=100)
   assert x > 2, "Greater than failed"
   assert 20 > x, "Greater than failed"
   assert y > x, "Greater than failed"


def test_ge():
   x = AutoDiff(name='x', val=10)
   y = AutoDiff(name='y', val=100)
   assert x >= 2, "Greater than or equal to failed"
   assert x >= 10, "Greater than or equal to failed"
   assert 20 >= x, "Greater than or equal to failed"
   assert 10 >= x, "Greater than or equal to failed"
   assert y >= x, "Greater than or equal to failed"

def test_lt():
   x = AutoDiff(name='x', val=10)
   y = AutoDiff(name='y', val=100)
   assert 2 < x, "Less than failed"
   assert x < 20, "Less than failed"
   assert x < y, "Less than failed"

def test_le():
   x = AutoDiff(name='x', val=10)
   y = AutoDiff(name='y', val=100)
   assert 2 <= x, "Less than or equal to failed"
   assert 10 <= x, "Less than or equal to failed"
   assert x <= 20, "Less than or equal to failed"
   assert x <= 10, 'Less than or equal to failed'
   assert x <= y, "Less than or equal to failed"

def test_ne():
   x = AutoDiff(name='x', val=10)
   y = AutoDiff(name='y', val=100)
   assert x != 11, "Not equal failed"
   assert 11 != x, "Not equal failed"
   assert x != y, "Not equal failed"

def test_modulo():
   x = AutoDiff(name='x', val=15)
   y = AutoDiff(name='y', val=12)
   z = AutoDiff(name='z', val=4)
   assert x % 2 == 1, "Modulo failed"
   assert y % 2 == 0, "Modulo failed"
   assert y % z == 0, "Modulo failed"
   assert x % z == 3, "Modulo failed"

# How to do rshift for scalars?
# How to do lshfit for scalars?

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

def test_len():
   x = AutoDiff(name='x', val=133)
   assert len(x) == 2, "Len failed"
   assert 2 == len(x), "Len failed"

def test_str():
   x = AutoDiff(name='x', val=2)
   assert str(x) == "{'val': 2, 'd_x': 1}", "Str failed"

#Didn't know how to test contains
# def test_contains():
#    x = AutoDiff(name='x', val=2)
#    assert contains(x) == 2, "Contains failed"

# Didn't know how to implement invert for scalar inputs
# def test_invert():
#    x = AutoDiff(name='x', val=3)
#    assert 1/3 = ~x, "Invert failed"
