from ARRRtomatic_diff import AutoDiffRev
from ARRRtomatic_diff import functions as ad
import numpy as np


def test_sin():
    x = AutoDiffRev(name='x', val=np.pi)
    y = AutoDiffRev(name='y', val=3 * np.pi)
    z = AutoDiffRev(name='z', val=-12)
    q = AutoDiffRev(name='q', val="string")
    assert ad.sin(x) == np.sin(np.pi), "Sine failed"
    assert ad.sin(x).get_gradient()[0][0] == -1, 'Sine failed'
    assert np.allclose(ad.sin(y).get_value(), ad.sin(x).get_value(), atol=1e-12) is True, "Sine failed"
    assert np.allclose(ad.sin(y).get_gradient()[0][0], ad.sin(x).get_gradient()[0][0], atol=1e-12) is True, "Sine failed"
    assert np.allclose(ad.sin(z).get_value(), 0.536572918, atol=1e-12) is True, "Sine failed"
    try:
        ad.sin(q)
    except TypeError:
        print("Caught error as expected")


def test_cos():
    x = AutoDiffRev(name='x', val=np.pi)
    y = AutoDiffRev(name='y', val=-np.pi)
    q = AutoDiffRev(name='q', val="string")
    assert ad.cos(x) == -1, "Cosine failed"
    assert np.allclose(ad.cos(x).get_gradient()[0][0], 0, atol=1e-12) is True, 'Cosine failed'
    assert ad.cos(y) == np.cos(-np.pi), "Cosine failed"
    assert np.allclose(ad.cos(y).get_gradient()[0][0], -np.sin(-np.pi), atol=1e-12) is True, "Cosine failed"
    try:
        ad.cos(q)
    except TypeError:
        print("Caught error as expected")


def test_tan():
    x = AutoDiffRev(name='x', val=np.pi)
    y = AutoDiffRev(name='y', val=np.pi / 2)
    q = AutoDiffRev(name='q', val="string")
    assert ad.tan(x) == np.tan(np.pi), "Tan failed"
    assert np.allclose(ad.tan(x).get_gradient()[0][0], (1 / np.cos(np.pi)) ** 2, atol=1e-12) is True, "Tan failed"
    assert ad.tan(y).get_value() == np.tan(np.pi/2), "Tan failed"
    try:
        ad.tan(q)
    except TypeError:
        print("Caught error as expected")


def test_csc():
    x = AutoDiffRev(name='x', val=np.pi / 2)
    assert np.allclose(ad.csc(x).get_value(), 1, atol=1e-12) is True, "Cosecant failed"
    assert np.allclose(ad.csc(x).get_gradient()[0][0], 0, atol=1e-12) is True, "Cosecant failed"


def test_sec():
    x = AutoDiffRev(name='x', val=0)
    assert np.allclose(ad.sec(x).get_value(), 1, atol=1e-12) is True, "Secant failed"
    assert np.allclose(ad.sec(x).get_gradient()[0][0], 0, atol=1e-12) is True, "Secant failed"


def test_cot():
    x = AutoDiffRev(name='x', val=np.pi / 2)
    assert np.allclose(ad.cot(x).get_value(), 0, atol=1e-12), "Cotangent failed"
    assert np.allclose(ad.cot(x).get_gradient()[0][0], -1, atol=1e-12), "Cotangent failed"


def test_asin():
    x = AutoDiffRev(name='x', val=0)
    assert np.allclose(ad.asin(x).get_value(), 0, atol=1e-12) is True, "Arcsin failed"
    assert np.allclose(ad.asin(x).get_gradient()[0][0], 1, atol=1e-12) is True, "Arcsin failed"


def test_acos():
    x = AutoDiffRev(name='x', val=0)
    assert np.allclose(ad.acos(x).get_value(), np.pi / 2, atol=1e-12) is True, 'Arccos failed'
    assert np.allclose(ad.acos(x).get_gradient()[0][0], -1, atol=1e-12) is True, "Arccos failed"


def test_atan():
    x = AutoDiffRev(name='x', val=np.pi / 2)
    assert np.allclose(ad.atan(x).get_value(), 1.00388482185388721414842, atol=1e-12) is True, "Arctan failed"
    assert np.allclose(ad.atan(x).get_gradient()[0][0], 0.2884004391420009, atol=1e-12) is True, "Arctan failed"


def test_acsc():
    x = AutoDiffRev(name='x', val=np.pi / 2)
    assert np.allclose(ad.acsc(x).get_value(), 0.69010709137453995200437, atol=1e-12) is True, "Arccosecant failed"
    assert np.allclose(ad.acsc(x).get_gradient()[0][0], -0.525539910519202993781, atol=1e-12) is True, "Arccosecant failed"


def test_asec():
    x = AutoDiffRev(name='x', val=2)
    assert np.allclose(ad.asec(x).get_value(), np.pi / 3, atol=1e-12) is True, "Arcsecant failed"
    assert np.allclose(ad.asec(x).get_gradient()[0][0], 1 / (2 * np.sqrt(3)), atol=1e-12) is True, "Arcsecant failed"


def test_acot():
    x = AutoDiffRev(name='x', val=1)
    assert np.allclose(ad.acot(x).get_value(), np.pi / 4, atol=1e-12) is True, "Arccotangent failed"
    assert np.allclose(ad.acot(x).get_gradient()[0][0], -0.5, atol=1e-12) is True, "Arccotangent failed"


def test_sinh():
    x = AutoDiffRev(name='x', val=-1)
    assert np.allclose(ad.sinh(x).get_value(), -1.17520119364380145688, atol=1e-12) is True, 'Sinh failed'
    assert np.allclose(ad.sinh(x).get_gradient()[0][0], 1.543080634815243778477905, atol=1e-12) is True, "Sinh failed"


def test_cosh():
    x = AutoDiffRev(name='x', val=np.pi)
    assert np.allclose(ad.cosh(x).get_value(), 11.591953275521520627751, atol=1e-12) is True, "Cosh failed"
    assert np.allclose(ad.cosh(x).get_gradient()[0][0], 11.54873935725774837797733, atol=1e-12) is True, "Cosh failed"


def test_tanh():
    x = AutoDiffRev(name='x', val=0.5)
    assert np.allclose(ad.tanh(x).get_value(), 0.462117, atol=1e-12) is True, "Tanh failed"
    assert np.allclose(ad.tanh(x).get_gradient()[0][0], 0.786448, atol=1e-12) is True, "Tanh failed"


def test_csch():
    x = AutoDiffRev(name='x', val=1)
    assert np.allclose(ad.csch(x).get_value(), 0.8509181282393215451, atol=1e-12) is True, "Csch failed"
    assert np.allclose(ad.csch(x).get_gradient()[0][0], -1.117285527449274171482, atol=1e-12) is True, "Csch failed"


def test_sech():
    x = AutoDiffRev(name='x', val=0)
    assert np.allclose(ad.sech(x).get_value(), 1, atol=1e-12) is True, "Sech failed"
    assert np.allclose(ad.sech(x).get_gradient()[0][0], 0, atol=1e-12) is True, "Sech failed"


def test_coth():
    x = AutoDiffRev(name='x', val=2)
    assert np.allclose(ad.coth(x).get_value(), 1.0373147207275480958778, atol=1e-12) is True, "Coth failed"
    assert np.allclose(ad.coth(x).get_gradient()[0][0], -0.07602182983807109925337, atol=1e-12) is True, "Coth failed"


def test_asinh():
    x = AutoDiffRev(name='x', val=np.pi / 2)
    assert np.allclose(ad.asinh(x).get_value(), 1.2334031175112170570, atol=1e-12) is True, "Asinh failed"
    assert np.allclose(ad.asinh(x).get_gradient()[0][0], 0.5370292721463150768, atol=1e-12) is True, "Asinh failed"


def test_acosh():
    x = AutoDiffRev(name='x', val=2)
    assert np.allclose(ad.acosh(x).get_value(), 1.316957896924816, atol=1e-12) is True, "Acosh failed"
    assert np.allclose(ad.acosh(x).get_gradient()[0][0], 1 / np.sqrt(3), atol=1e-12) is True, "Acosh failed"


def test_atanh():
    x = AutoDiffRev(name='x', val=0)
    assert np.allclose(ad.atanh(x).get_value(), 0, atol=1e-12) is True, "Atanh failed"
    assert np.allclose(ad.atanh(x).get_gradient()[0][0], 1, atol=1e-12) is True, "Atanh failed"


def test_acsch():
    x = AutoDiffRev(name='x', val=1)
    assert np.allclose(ad.acsch(x).get_value(), 0.881373587019543025232, atol=1e-12) is True, "Acsch failed"
    assert np.allclose(ad.acsch(x).get_gradient()[0][0], -1 / np.sqrt(2), atol=1e-12) is True, "Acsch failed"


def test_asech():
    x = AutoDiffRev(name='x', val=0.2)
    assert np.allclose(ad.asech(x).get_value(), 2.29243, atol=1e-12) is True, "Asech failed"
    assert np.allclose(ad.asech(x).get_gradient()[0][0], -5.10310, atol=1e-12) is True, "Asech failed"


def test_acoth():
    x = AutoDiffRev(name='x', val=2)
    assert np.allclose(ad.acoth(x).get_value(), 0.5493061443340, atol=1e-12) is True, "Acoth failed"
    assert np.allclose(ad.acoth(x).get_gradient()[0][0], -1 / 3, atol=1e-12) is True, "Acoth failed"
