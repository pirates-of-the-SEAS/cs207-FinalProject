
from ARRRtomatic_diff import AutoDiff

def test_add():
   # from ARRRtomatic_diff.auto_diff import AutoDiff
   x = AutoDiff(name='x', val=2)
   y = AutoDiff(name='y', val=-5)
   # print(type(x+1))
   assert (x + 1).trace['val'] == 3
   assert (x + 1).trace['d_x'] == 1
   assert (1 + x).trace['val'] == 3
   assert (1 + x).trace['d_x'] == 1

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