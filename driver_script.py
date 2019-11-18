import math
import numpy as np

from ARRRtomatic_diff import AutoDiff
from ARRRtomatic_diff.functions import sin, exp

x = AutoDiff(trace={'val': 3, 'd_x': 4, 'd_y': 2}, name=set(('x', 'y')))

print(x)
