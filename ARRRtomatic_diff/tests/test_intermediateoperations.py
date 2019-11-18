from ARRRtomatic_diff import AutoDiff
from ARRRtomatic_diff import functions as ad
import numpy as np
import warnings


def test_sqrt():
    x = AutoDiff(name='x', val=16)
    y = AutoDiff(name='y', val=0)
    z = AutoDiff(name='z', val=-4)
    assert ad.sqrt(x) == 4, "Square root failed"
    assert ad.sqrt(x).trace['d_x'] == 1 / (2 * np.sqrt(16)), "Square root failed"
    # Gives runtime warning
    # assert ad.sqrt(y) == 0, "Square root failed"

    # Should give 'nan' or handle imaginary, but outputs nan that does not equate to np.nan
    # assert ad.sqrt(z).trace['val'] == np.nan, "Square root failed"


def test_Euler():
    x = AutoDiff(name='x', val=3)
    y = AutoDiff(name='y', val=0)
    z = AutoDiff(name='z', val=-0.34020)
    assert ad.exp(x) == np.exp(3), "Euler's number failed"
    assert ad.exp(x).trace['d_x'] == np.exp(3), "Euler's number failed"
    # Returns 20.085536923187668 instead of 1
    # assert ad.exp(x) == np.exp(0), "Euler's number failed"
    assert ad.exp(y).trace['d_y'] == np.exp(0), "Euler's number failed"
    assert ad.exp(z) == np.exp(-0.34020), "Euler's number failed"
    assert ad.exp(z).trace['d_z'] == np.exp(-0.34020), "Euler's number failed"


def test_log():
    x = AutoDiff(name='x', val=4)
    y = AutoDiff(name='y', val=0)
    z = AutoDiff(name='z', val=-2)
    assert np.allclose((ad.log(x)).trace['val'], np.log(4), atol=1e-12) == True, 'Log failed'
    assert ad.log(x).trace['d_x'] == 1 / 4, 'Log failed'

    # Doesn't throw error, but I can't catch RuntimeWarning"
    # with warnings.catch_warnings():
    #     warnings.filterwarnings(action='error', message="RuntimeWarning")
    #     try:
    #         ad.log(z).trace['val']
    #     except Warning as e:
    #         print("Caught error as expected")

    # Won't catch RuntimeWarning
    # try:
    #     ad.log(z)
    # except RuntimeWarning:
    #     print("Caught Warning as expected")
