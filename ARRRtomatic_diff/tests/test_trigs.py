from ARRRtomatic_diff import AutoDiff
from ARRRtomatic_diff import functions as ad
import numpy as np

def test_sin():
    x = AutoDiff(name='x', val=np.pi)
    assert ad.sin(x) == np.sin(np.pi), "Sine failed"
    assert ad.sin(x).trace['d_x'] == -1, 'Sine failed'

def test_cos():
    x = AutoDiff(name='x', val=np.pi)
    y = AutoDiff(name='y', val=-np.pi)
    assert (ad.cos(x)) == -1, "Cosine failed"
    assert np.allclose((ad.cos(x)).trace['d_x'], 0, atol=1e-12) is True, 'Cosine failed'
    assert ad.cos(y) == np.cos(-np.pi), "Cosine failed"
    assert np.allclose(ad.cos(y).trace['d_y'], -np.sin(-np.pi), atol=1e-12) is True, "Cosine failed"

def test_tan():
    x = AutoDiff(name='x', val=np.pi)
    assert ad.tan(x) == np.tan(np.pi), "Tan failed"
    assert np.allclose(ad.tan(x).trace['d_x'], (1/np.cos(np.pi))**2, atol=1e-12) is True, "Tan failed"

def test_csc():
    x = AutoDiff(name='x', val=np.pi/2)
    assert ad.csc(x) == 1, "Cosecant failed"
    # y = AutoDiff(name='y', val=np.pi)
    # print(ad.csc(np.pi))

def test_sec():
    x = AutoDiff(name='x', val=0)
    assert ad.sec(x) == 1, "Secant failed"

# def test_cot():
    # x = AutoDiff(name='x', val=np.pi/4)
    # assert np.allclose(ad.cot(x), 1, atol=1e-12), "Cotangent failed"

def test_asin():
    x = AutoDiff(name='x', val=0)
    assert ad.asin(x) == 0, "Arcsin failed"

#
# def test_acos():
#     x = AutoDiff(name='x', val=0)
#     assert ad.arccos(x) == np.pi/2, 'Arccos failed'
#
# def test_atan():
#
# def test_acsc():
#
# def test_asec():
#
# def test_acot():
#
# def test_sinh():
#     x = AutoDiff(name='x', val=0)
#     y = AutoDiff(name='y', val=np.pi/2)
#     assert ad.sinh(x) == 0, 'Sinh failed'
#     assert np.allclose((ad.sinh(y)).trace['val'], 2.30129890230729487346, atol=1e-12) == True, 'Sinh failed'
#
# def test_cosh():
#
# def test_tanh():
#
# def test_csch():
#
# def test_sech():
#
# def test_coth():
#
# def test_asinh():
#
# def test_acosh():
#
# def test_atanh():
#
# def test_acsch():
#
# def test_asech():
#
# def test_acoth():
