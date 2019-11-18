import math
import numpy as np

from ARRRtomatic_diff import AutoDiff
from ARRRtomatic_diff.functions import sin, exp

y = AutoDiff(name='y', val=0)
z = AutoDiff(name='z', val=-2)
assert (z**2) == 4, "Exponentiation failed"
assert (z**3) == -8, "Exponentiation failed"
assert (y**2) == 0, "Exponentiation failed"
