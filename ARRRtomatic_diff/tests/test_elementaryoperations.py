from ARRRtomatic_diff import AutoDiff

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
   assert x > 2, "Greater than failed"
   assert 20 > x, "Greater than failed"

def test_lt():
   x = AutoDiff(name='x', val=10)
   assert 2 < x, "Less than failed"
   assert x < 20, "Less than failed"