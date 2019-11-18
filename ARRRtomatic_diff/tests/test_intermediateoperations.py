from ARRRtomatic_diff import AutoDiff
from ARRRtomatic_diff import functions as ad
import numpy as np
import warnings

def test_log():
    x = AutoDiff(name='x', val=4)
    y = AutoDiff(name='y', val=0)
    z = AutoDiff(name='z', val=-2)
    assert np.allclose((ad.log(x)).trace['val'], np.log(4), atol=1e-12) == True, 'Log failed'
    assert ad.log(x).trace['d_x'] == 1/4, 'Log failed'
    # try:
    #     ad.log(y)
    # except TypeError:
    #     print("Caught TypeError as expected")

    # with warnings.catch_warnings():
    #     warnings.filterwarnings('error')
    #     try:
    #         ad.log(z).trace['val']
    #     except Warning as e:
    #         print('error found:', e)
    # try:
    #     ad.log(z)
    # except RuntimeWarning:
    #     print("Caught Warning as expected")

#
# def test_Euler():
#
# def test_sqrt():
