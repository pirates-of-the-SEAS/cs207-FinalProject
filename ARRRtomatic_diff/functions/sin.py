"""
"""

import numpy as np
from auto_diff import AutoDiff

def sin(x):

    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.sin(updated_trace['val'])

        for var in named_variables:
            print(updated_trace[f'd_{var}'])
            print(trace['val'])
            updated_trace[f'd_{var}'] = np.cos(trace['val']) * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        return np.sin(x)


if __name__ == '__main__':
    pass
