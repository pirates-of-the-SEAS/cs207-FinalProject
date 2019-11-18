from ARRRtomatic_diff import AutoDiff

def test_add():
   x = AutoDiff(name='x', val=2)
   y = AutoDiff(name='y', val=-5)
   z = AutoDiff(name='z', val=0)
   assert (x + 1) == 3, 'Addition failed'
   assert (x + 1).trace['d_x'] == 1, 'Addition failed'
   assert (1 + x) == 3, 'Addition failed'
   assert (1 + x).trace['d_x'] == 1, 'Addition failed'
   assert (y + x) == -3, 'Addition failed'
   assert (y + x).trace['d_x'] == 1, 'Addition failed'
   assert (y + x).trace['d_y'] == 1, 'Addition failed'
   assert (z + z) == 0, 'Addition failed'
   assert (z + z).trace['d_z'] == 2, 'Addition failed'

def test_subtract():
   x = AutoDiff(name='x', val=9)
   y = AutoDiff(name='y', val=-5)
   z = AutoDiff(name='z', val=0)
   assert (x - 1) == 8, 'Subtraction failed'
   assert (x - 1).trace['d_x'] == 1, 'Subtraction failed'
   assert (1 - x) == -8, 'Subtraction failed'
   assert (1 - x).trace['d_x'] == -1, 'Subtraction failed'
   assert (y - x) == -14, 'Subtraction failed'
   assert (y - x).trace['d_x'] == -1, 'Subtraction failed'
   assert (y - x).trace['d_y'] == 1, 'Subtraction failed'
   assert (z + z) == 0, 'Subtraction failed'
   assert (z + z).trace['d_z'] == 2, 'Subtraction failed'

def test_multiply():
   x = AutoDiff(name='x', val=6)
   y = AutoDiff(name='y', val=-5)
   z = AutoDiff(name='z', val=0)
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

def test_divide():
   x = AutoDiff(name='x', val=6)
   y = AutoDiff(name='y', val=-12)
   z = AutoDiff(name='z', val=0)
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

def test_exponentiation():
    x = AutoDiff(name='x', val=3)
    y = AutoDiff(name='')
    assert (x**2) == 9, "Exponentiation failed"
    assert (x**0) == 1, "Exponentiation failed"
    assert (x**-2) == (1/9), "Exponentiation failed"

