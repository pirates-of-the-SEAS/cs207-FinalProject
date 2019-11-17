
from ARRRtomatic_diff import AutoDiff

def test_add():
   # from ARRRtomatic_diff.auto_diff import AutoDiff
   x = AutoDiff(name='x', val=2)
   y = AutoDiff(name='y', val=-5)
   z = AutoDiff(name='z', val=0)
   assert (x + 1).trace['val'] == 3, 'Addition failed'
   assert (x + 1).trace['d_x'] == 1, 'Addition failed'
   assert (1 + x).trace['val'] == 3, 'Addition failed'
   assert (1 + x).trace['d_x'] == 1, 'Addition failed'
   assert (y + x).trace['val'] == -3, 'Addition failed'
   assert (y + x).trace['d_x'] == 1, 'Addition failed'
   assert (y + x).trace['d_y'] == 1, 'Addition failed'
   assert (z + z).trace['val'] == 0, 'Addition failed'
   assert (z + z).trace['d_z'] == 2, 'Addition failed'

def test_subtract():
   # from ARRRtomatic_diff.auto_diff import AutoDiff
   x = AutoDiff(name='x', val=9)
   y = AutoDiff(name='y', val=-5)
   z = AutoDiff(name='z', val=0)
   assert (x - 1).trace['val'] == 8, 'Subtraction failed'
   assert (x - 1).trace['d_x'] == 1, 'Subtraction failed'
   assert (1 - x).trace['val'] == -8, 'Subtraction failed'
   assert (1 - x).trace['d_x'] == 1, 'Subtraction failed'
   assert (y - x).trace['val'] == -14, 'Subtraction failed'
   assert (y - x).trace['d_x'] == 1, 'Subtraction failed'
   assert (y - x).trace['d_y'] == 1, 'Subtraction failed'
   assert (z + z).trace['val'] == 0, 'Subtraction failed'
   assert (z + z).trace['d_z'] == 2, 'Subtraction failed'

def test_multiply():
   # from ARRRtomatic_diff.auto_diff import AutoDiff
   x = AutoDiff(name='x', val=6)
   y = AutoDiff(name='y', val=-5)
   z = AutoDiff(name='z', val=0)
   assert (x * 2).trace['val'] == 12, 'Multiplication failed'
   assert (x * 1).trace['d_x'] == 2, 'Multiplication failed'
   assert (2 * x).trace['val'] == 12, 'Multiplication failed'
   assert (2 * x).trace['d_x'] == 2, 'Multiplication failed'
   assert (y * x).trace['val'] == -30, 'Multiplication failed'
   assert (y * x).trace['d_x'] == -5, 'Multiplication failed'
   assert (y * x).trace['d_y'] == 6, 'Multiplication failed'
   assert (z * z).trace['val'] == 0, 'Multiplication failed'
   assert (z * z).trace['d_z'] == 0, 'Multiplication failed'

# def test_add():
#    # from ARRRtomatic_diff.auto_diff import AutoDiff
#    x = AutoDiff(name='x', val=2)
#    y = AutoDiff(name='y', val=-5)
#    # print(type(x+1))
#    assert (x + 1).trace['val'] == 3
#    assert (x + 1).trace['d_x'] == 1
#    assert (1 + x).trace['val'] == 3
#    assert (1 + x).trace['d_x'] == 1

# def test_addition():
#     x = AutoDiff(name='a0', val=3)
#     y = AutoDiff(name='a1', val=0)
#     z = AutoDiff(name='a2', val=-200)
#
#     assert x + x == 6, 'Addition failed'
#     assert y + y == 0, 'Addition failed'
#     assert z + z == -400, 'Addition failed'

    # addition
    # print("Addition")
    # print(1 + x)
    # print(x + 1)
    #
    # print(x + x)
    #
    # print(x + y)
    # print(y + x)