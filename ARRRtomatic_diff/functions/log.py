"""
"""

import numpy as np
from ARRRtomatic_diff import AutoDiff

def log(x):

    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.log(updated_trace['val'])

        for var in named_variables:
            updated_trace[f'd_{var}'] = 1/trace['val'] * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        # return real part
        return np.log(x)


if __name__ == '__main__':
    pass
