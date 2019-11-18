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
    assert np.allclose(ad.csc(x).trace['val'], 1, atol=1e-12), "Cosecant failed"
    # y = AutoDiff(name='y', val=np.pi)
    # print(ad.csc(np.pi))

def test_sec():
    x = AutoDiff(name='x', val=0)
    assert np.allclose(ad.sec(x).trace['val'], 1, atol=1e-12) is True, "Secant failed"
    # assert np.allclose(ad.sec(x).trace['d_x'], 1, atol=1e-12) is True, "Secant failed"

# def test_cot():
    # x = AutoDiff(name='x', val=np.pi/4)
    # assert np.allclose(ad.cot(x), 1, atol=1e-12), "Cotangent failed"

def test_asin():
    x = AutoDiff(name='x', val=0)
    assert np.allclose(ad.asin(x).trace['val'], 0, atol=1e-12) is True, "Arcsin failed"

def test_acos():
    x = AutoDiff(name='x', val=0)
    assert np.allclose(ad.arccos(x).trace['val'], np.pi/2, atol=1e-12) is True, 'Arccos failed'

def test_atan():
    x = AutoDiff(name='x', val=np.pi/2)
    assert np.allclose(ad.atan(x).trace['val'], 1.00388482185388721414842, atol=1e-12) is True, "Arctan failed"

def test_acsc():
    x = AutoDiff(name='x', val=np.pi/2)
    assert np.allclose(ad.acsc(x).trace['val'], 0.69010709137453995200437, atol=1e-12) is True, "Arccosecant failed"

def test_asec():
    x = AutoDiff(name='x', val=2)
    assert np.allclose(ad.asec(x).trace['val'], np.pi/3, atol=1e-12) is True, "Arcsecant failed"

def test_acot():
    x = AutoDiff(name='x', val=1)
    assert np.allclose(ad.acot(x).trace['val'], np.pi/4, atol=1e-12) is True, "Arccotangent failed"

def test_sinh():
    x = AutoDiff(name='x', val=-1)
    assert np.allclose(ad.sinh(x).trace['val'], -1.17520119364380145688, atol=1e-12) is True, 'Sinh failed'
    assert np.allclose(ad.sinh(x).trace['d_x'], 1.543080634815243778477905, atol=1e-12) is True, "Sinh failed"

def test_cosh():
    x = AutoDiff(name='x', val=np.pi)
    assert np.allclose(ad.cosh(x).trace['val'], 11.591953275521520627751, atol=1e-12) is True, "Cosh failed"
    assert np.allclose(ad.cosh(x).trace['d_x'], 11.54873935725774837797733, atol=1e-12) is True, "Cosh failed"

def test_tanh():
    x = AutoDiff(name='x', val=0.5)
    assert np.allclose(ad.tanh(x).trace['val'], 0.462117, atol=1e-12) is True, "Tanh failed"
    assert np.allclose(ad.tanh(x).trace['d_x'], 0.786448, atol=1e-12) is True, "Tanh failed"

def test_csch():
    x = AutoDiff(name='x', val=1)
    assert np.allclose(ad.csch(x).trace['val'], 0.8509181282393215451, atol=1e-12) is True, "Csch failed"
    assert np.allclose(ad.csch(x).trace['d_x'], -1.117285527449274171482, atol=1e-12) is True, "Csch failed"

def test_sech():
    x = AutoDiff(name='x', val=0)
    assert np.allclose(ad.sech(x).trace['val'], 1, atol=1e-12) is True, "Sech failed"
    assert np.allclose(ad.sech(x).trace['d_x'], 0, atol=1e-12) is True, "Sech failed"

def test_coth():
    x = AutoDiff(name='x', val=2)
    assert np.allclose(ad.coth(x).trace['val'], 1.0373147207275480958778, atol=1e-12) is True, "Coth failed"
    assert np.allclose(ad.coth(x).trace['d_x'], -0.07602182983807109925337, atol=1e-12) is True, "Coth failed"

def test_asinh():
    x = AutoDiff(name='x', val=np.pi/2)
    assert np.allclose(ad.asinh(x).trace['val'], 1.2334031175112170570, atol=1e-12) is True, "Asinh failed"
    assert np.allclose(ad.asinh(x).trace['d_x'], 0.5370292721463150768, atol=1e-12) is True, "Asinh failed"

def test_acosh():
    x = AutoDiff(name='x', val=2)
    assert np.allclose(ad.acosh(x).trace['val'], 1.316957896924816, atol=1e-12) is True, "Acosh failed"
    assert np.allclose(ad.acosh(x).trace['d_x'], 1/np.sqrt(3), atol=1e-12) is True, "Acosh failed"

def test_atanh():
    x = AutoDiff(name='x', val=0)
    assert np.allclose(ad.atanh(x).trace['val'], 0, atol=1e-12) is True, "Atanh failed"
    assert np.allclose(ad.atanh(x).trace['d_x'], 1, atol=1e-12) is True, "Atanh failed"

def test_acsch():
    x = AutoDiff(name='x', val=1)
    assert np.allclose(ad.acsch(x).trace['val'], 0.881373587019543025232, atol=1e-12) is True, "Acsch failed"
    assert np.allclose(ad.acsch(x).trace['d_x'], -1/np.sqrt(2), atol=1e-12) is True, "Acsch failed"

def test_asech():
    x = AutoDiff(name='x', val=0.2)
    assert np.allclose(ad.asech(x).trace['val'], 2.29243, atol=1e-12) is True, "Asech failed"
    assert np.allclose(ad.asech(x).trace['d_x'], -5.10310, atol=1e-12) is True, "Asech failed"

def test_acoth():
    x = AutoDiff(name='x', val=2)
    assert np.allclose(ad.acoth(x).trace['val'], 0.5493061443340, atol=1e-12) is True, "Acoth failed"
    assert np.allclose(ad.acoth(x).trace['d_x'], -1/3, atol=1e-12) is True, "Acoth failed"
