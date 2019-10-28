"""
"""

import numpy as np
from auto_diff import AutoDiff


def exp(x):

    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.exp(updated_trace['val'])

        
        for var in named_variables:
            print(var)

            updated_trace[f'd_{var}'] = updated_trace['val'] * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        return np.exp(x)


if __name__ == '__main__':
    pass
