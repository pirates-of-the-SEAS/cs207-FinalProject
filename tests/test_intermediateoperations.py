from ARRRtomatic_diff import AutoDiff
from ARRRtomatic_diff import functions as ad
import numpy as np


def test_sqrt():
    x = AutoDiff(name='x', val=16)
    y = AutoDiff(name='y', val=0)
    z = AutoDiff(name='z', val=-4)
    assert ad.sqrt(x) == 4, "Square root failed"
    assert ad.sqrt(x).trace['d_x'] == 1 / (2 * np.sqrt(16)), "Square root failed"
    try:
        ad.sqrt(y).trace['val']
    except ValueError:
        print("Caught error as expected")
    try:
        ad.sqrt(z).trace['val']
    except ValueError:
        print("Caught error as expected")


def test_euler():
    x = AutoDiff(name='x', val=3)
    y = AutoDiff(name='y', val=0)
    z = AutoDiff(name='z', val=-0.34020)

    assert np.allclose(ad.exp(x).trace['val'], np.exp(3)), "Euler's number failed"
    assert np.allclose(ad.exp(x).trace['d_x'], np.exp(3)), "Euler's number failed"
    assert np.allclose(ad.exp(y).trace['val'], np.exp(0)), "Euler's number failed"
    assert np.allclose(ad.exp(y).trace['d_y'], np.exp(0)), "Euler's number failed"
    assert np.allclose(ad.exp(z).trace['val'], np.exp(-0.34020)), "Euler's number failed"
    assert np.allclose(ad.exp(z).trace['d_z'], np.exp(-0.34020)), "Euler's number failed"


def test_log():
    x = AutoDiff(name='x', val=4)
    y = AutoDiff(name='y', val=0)
    z = AutoDiff(name='z', val=-2)
    assert np.allclose(ad.log(x).trace['val'], np.log(4), atol=1e-12) == True, 'Log failed'
    assert ad.log(x).trace['d_x'] == 1 / 4, 'Log failed'
    try:
        ad.log(y).trace['d_y']
    except ValueError:
        print("Caught error as expected")
    try:
        ad.log(z)
    except ValueError:
        print("Caught error as expected")


def test_composition():
    x = AutoDiff(name='x', val=4)
    assert np.allclose(ad.sin(ad.exp(x) ** 2).trace['val'], 0.4017629715192812,
                       atol=1e-12) is True, "Composition failed"
    assert np.allclose(ad.sin(ad.exp(x) ** 2).trace['d_x'], -5459.586962682745,
                       atol=1e-12) is True, "Composition failed"
    assert np.allclose(ad.exp(ad.sin(x) ** 2).trace['val'], 1.7731365081918968985940,
                       atol=1e-12) is True, "Composition failed"
    assert np.allclose(ad.exp(ad.sin(x) ** 2).trace['d_x'], 1.75426722676864073577,
                       atol=1e-12) is True, "Composition failed"


def test_root():
    x = AutoDiff(name='x', val=144)
    result = ad.root(x, 4)
    assert np.allclose(result.trace['val'], 144**(1/4)) is True, "Root failed"
    assert np.allclose(result.trace['d_x'], 0.006014065304058602) is True, "Root failed"


def test_logistic():
    x = AutoDiff(name='x', val=3)
    assert np.allclose(ad.logistic(x).trace['val'], 1/(1 + np.exp(-3))) is True, "Logistic failed"
    assert np.allclose(ad.logistic(x).trace['d_x'], 0.045176659730912144) is True, "Logistic failed"