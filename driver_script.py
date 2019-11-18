import math
import numpy as np

from ARRRtomatic_diff import AutoDiff
from ARRRtomatic_diff.functions import sin, exp

x = AutoDiff(name='x', val=0)

print(x**5)

