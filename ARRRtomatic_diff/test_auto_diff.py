from .auto_diff import AutoDiff


x = AutoDiff(name='x', val=2)
y = AutoDiff(name='y', val=-5)

def test_add():
     
    assert (x+1)['val'] ==3
    assert (x+1)['d_x'] ==1
    assert (1+x)['val'] ==3
    assert (1+x)['d_x'] ==1



