from ARRRtomatic_diff import AutoDiff
import numpy as np

def test_cos():
    x = AutoDiff(name='x', val=np.pi)
    assert (np.cos(x.trace['val'])) == -1, "Cosine failed"