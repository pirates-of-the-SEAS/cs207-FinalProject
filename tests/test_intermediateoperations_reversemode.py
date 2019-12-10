from ARRRtomatic_diff import AutoDiffRev
from ARRRtomatic_diff import functions as ad
import numpy as np


def test_sqrt():
    x = AutoDiffRev(name='x', val=16)
    y = AutoDiffRev(name='y', val=0)
    z = AutoDiffRev(name='z', val=-4)
    assert ad.sqrt(x) == 4, "Square root failed"
    g, _ = ad.sqrt(x).get_gradient()
    assert g == [1 / (2 * np.sqrt(16))], "Square root failed"
    try:
        ad.sqrt(y).get_gradient()
    except ValueError:
        print("Caught error as expected")
    try:
        ad.sqrt(z).get_gradient()
    except ValueError:
        print("Caught error as expected")


def test_Euler():
    x = AutoDiffRev(name='x', val=3)
    y = AutoDiffRev(name='y', val=0)
    z = AutoDiffRev(name='z', val=-0.34020)
    assert np.allclose(ad.exp(x).get_value(), np.exp(3)), "Euler's number failed"
    g, _ = ad.exp(x).get_gradient()
    assert np.allclose(g, [np.exp(3)]), "Euler's number failed"
    assert np.allclose(ad.exp(y).get_value(), np.exp(0)), "Euler's number failed"
    g, _ = ad.exp(y).get_gradient()
    assert np.allclose(g, [np.exp(0)]), "Euler's number failed"
    assert np.allclose(ad.exp(z).get_value(), np.exp(-0.34020)), "Euler's number failed"
    g, _ = ad.exp(z).get_gradient()
    assert np.allclose(g, np.exp(-0.34020)), "Euler's number failed"


def test_log():
    x = AutoDiffRev(name='x', val=4)
    y = AutoDiffRev(name='y', val=0)
    z = AutoDiffRev(name='z', val=-2)
    assert np.allclose(ad.log(x).get_value(), np.log(4), atol=1e-12) == True, 'Log failed'
    g, _ = ad.log(x).get_gradient()
    assert g == 1 / 4, 'Log failed'
    try:
        ad.log(y).get_gradient()
    except ValueError:
        print("Caught error as expected")
    try:
        ad.log(z)
    except ValueError:
        print("Caught error as expected")


def test_composition():
    x = AutoDiffRev(name='x', val=4)
    assert np.allclose(ad.sin(ad.exp(x) ** 2).get_value(), 0.4017629715192812,
                       atol=1e-12) is True, "Composition failed"
    g, _ = ad.sin(ad.exp(x) ** 2).get_gradient()
    assert np.allclose(g, -5459.586962682745,
                       atol=1e-12) is True, "Composition failed"
    assert np.allclose(ad.exp(ad.sin(x) ** 2).get_value(), 1.7731365081918968985940,
                       atol=1e-12) is True, "Composition failed"
    g, _ = ad.exp(ad.sin(x) ** 2).get_gradient()
    assert np.allclose(g, 1.75426722676864073577,
                       atol=1e-12) is True, "Composition failed"
