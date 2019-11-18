from ARRRtomatic_diff import AutoDiff
from ARRRtomatic_diff import functions as ad
import numpy as np

def test_cos():
    x = AutoDiff(name='x', val=np.pi)
    assert (ad.cos(x)) == -1, "Cosine failed"
    assert np.allclose((ad.cos(x)).trace['d_x'], 0, atol=1e-12) == True, 'Cosine failed'

def test_sin():
    x = AutoDiff(name='x', val=np.pi)
    assert np.allclose((ad.sin(x)).trace['val'], 0, atol=1e-12) == True, 'Sine failed'
    assert ad.sin(x).trace['d_x'] == -1, 'Sine failed'

def test_arccos():
    x = AutoDiff(name='x', val=0)
    assert ad.arccos(x) == np.pi/2, 'Arccos failed'

def test_sinh():
    x = AutoDiff(name='x', val=0)
    y = AutoDiff(name='y', val=np.pi/2)
    assert ad.sinh(x) == 0, 'Sinh failed'
    assert np.allclose((ad.sinh(y)).trace['val'], 2.30129890230729487346, atol=1e-12) == True, 'Sinh failed'