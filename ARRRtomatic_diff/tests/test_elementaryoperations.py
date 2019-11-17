
from ARRRtomatic_diff import AutoDiff

def test_addition():
    x = AutoDiff(name='a0', val=3)
    y = AutoDiff(name='a1', val=0)
    z = AutoDiff(name='a2', val=-200)

    assert x + x == 3, 'Addition failed'
    assert y + y == 0, 'Addition failed'
    assert z + z == -400, 'Addition failed'

    # addition
    # print("Addition")
    # print(1 + x)
    # print(x + 1)
    #
    # print(x + x)
    #
    # print(x + y)
    # print(y + x)