from ARRRtomatic_diff import AutoDiff
from ARRRtomatic_diff import functions as ad
import numpy as np

def test_cos():
    x = AutoDiff(name='x', val=np.pi)
    assert (ad.cos(x)) == -1, "New cosine failed"