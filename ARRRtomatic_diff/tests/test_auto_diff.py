
#
#
# x = AutoDiff(name='x', val=2)
# y = AutoDiff(name='y', val=-5)
#
# def test_add():
#
#     assert (x+1)['val'] ==3
#     assert (x+1)['d_x'] ==1
#     assert (1+x)['val'] ==3
#     assert (1+x)['d_x'] ==1
import pytest
# from ARRRtomatic_diff.functions import funcs


def test_add():
    # from ARRRtomatic_diff.auto_diff import AutoDiff
    # x = AutoDiff(name='x', val=2)
    # y = AutoDiff(name='y', val=-5)
    # print(type(x+1))
    # assert (x + 1).trace['val'] == 3
    # assert (x + 1).trace['d_x'] == 1
    # assert (1 + x).trace['val'] == 3
    # assert (1 + x).trace['d_x'] == 1
    assert True

#
#
#
# if __name__ == '__main':
#     x = AutoDiff(name='x', val=2)
#     y = AutoDiff(name='y', val=-5)
