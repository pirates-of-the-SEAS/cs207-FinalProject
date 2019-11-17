import numpy as np

from .. import AutoDiff


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

def sqrt(x):

    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.sqrt(updated_trace['val'])

        for var in named_variables:
            updated_trace[f'd_{var}'] = 1/2 * 1/np.sqrt(trace['val']) * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        # return real part
        return np.sqrt(x)

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


def csc(x):
    return 1/sin(x)

def sec(x):
    return 1/cos(x)

def cot(x):
    return 1/tan(x)

def asin(x):
    return arcsin(x)

def arcsin(x):
    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.arcsin(updated_trace['val'])

        for var in named_variables:
            print(updated_trace[f'd_{var}'])
            print(trace['val'])
            updated_trace[f'd_{var}'] = 1/np.sqrt(1 - trace['val']**2) * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        return np.arcsin(x)

def acos(x):
    return arccos(x)

def arccos(x):
    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.arccos(updated_trace['val'])

        for var in named_variables:
            print(updated_trace[f'd_{var}'])
            print(trace['val'])
            updated_trace[f'd_{var}'] = -1/np.sqrt(1 - trace['val']**2) * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        return np.arccosn(x)

def atan(x):
    return arctan(x)

def arctan(x):
    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.arctan(updated_trace['val'])

        for var in named_variables:
            print(updated_trace[f'd_{var}'])
            print(trace['val'])
            updated_trace[f'd_{var}'] = 1/(1 + trace['val']**2) * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        return np.arctan(x)

def acsc(x):
    return arccsc(x)

def arccsc(x):
    return asin(1/x)

def asec(x):
    return arcsec(x)

def arcsec(x):
    return acos(1/x)

def acot(x):
    return arccot(x)

def arccot(x):
    return atan(1/x)




def sinh(x):
    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.sinh(trace['val'])

        for var in named_variables:
            updated_trace[f'd_{var}'] = np.cosh(x) * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        return np.sinh(x)

def cosh(x):
    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.cosh(trace['val'])

        for var in named_variables:
            updated_trace[f'd_{var}'] = np.sinh(x) * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        return np.cosh(x)


def tanh(x):
    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.tanh(trace['val'])

        for var in named_variables:
            updated_trace[f'd_{var}'] = 1/np.cosh(x)**2 * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        return np.tanh(x)


def csch(x):
    return 1/sinh(x)

def sech(x):
    return 1/cosh(x)

def coth(x):
    return 1/tanh(x)



def asinh(x):
    return arcsinh(x)

def arcsinh(x):
    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.arcsinh(updated_trace['val'])

        for var in named_variables:
            print(updated_trace[f'd_{var}'])
            print(trace['val'])
            updated_trace[f'd_{var}'] = 1/np.sqrt(1 + trace['val']**2) * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        return np.arcsinh(x)

def acosh(x):
    return arccosh(x)

def arccosh(x):
    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.arccosh(updated_trace['val'])

        for var in named_variables:
            print(updated_trace[f'd_{var}'])
            print(trace['val'])
            updated_trace[f'd_{var}'] = 1/np.sqrt(1 + trace['val']) * 1/np.sqrt(trace['val'] - 1) * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        return np.arccosh(x)    

def atanh(x):
    return arctanh(x)

def arctanh(x):
    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        updated_trace = {}

        updated_trace.update(trace)
        updated_trace['val'] = np.arccosh(updated_trace['val'])

        for var in named_variables:
            print(updated_trace[f'd_{var}'])
            print(trace['val'])
            updated_trace[f'd_{var}'] = 1/(1 - trace['val1'**2]) * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                            trace=updated_trace)
    except:
        return np.arccosh(x)    

def acsch(x):
    return arccsch(x)

def arccsch(x):
    return asinh(1/x)

def asech(x):
    return arcsech(x)

def arcsech(x):
    return acosh(1/x)

def acoth(x):
    return arccoth(x)

def arccoth(x):
    return atanh(1/x)











