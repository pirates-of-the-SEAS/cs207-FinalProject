import numpy as np
from ARRRtomatic_diff.auto_diff import AutoDiff

def cos(x):

    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.cos(trace['val'])

        for var in named_variables:
            updated_trace[f'd_{var}'] = -1*np.sin(trace['val']) * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        return np.cos(x)


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

def tan(x):

    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.tan(updated_trace['val'])

        for var in named_variables:
            updated_trace[f'd_{var}'] = 1/(np.cos(trace['val']))**2 * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        return np.tan(x)