import math
import numpy as np

from ARRRtomatic_diff import AutoDiff
from ARRRtomatic_diff.functions import *

x = AutoDiff(name='x', val=np.pi)

print(cos(x) == -1)


