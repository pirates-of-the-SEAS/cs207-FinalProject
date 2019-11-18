import numpy as np

from .. import AutoDiff

def __update_unary(x, operation, doperation):
    """Updates an AutoDiff object with a unary operation or simply
        performs the unary operation on a numeric if it's supplied
        Not to be used externally.
    
    INPUTS
    =======
    x: The AutoDiff object or numeric. Uses ducktyping.
    operation: function, the unary operation, which is used to uodate the value 
    doperation: the derivative of the unary operation, which is used to update
                the gradients
    
    RETURNS
    ========
    An AutoDiff object whose value and gradient have been updated by the
     unary operation
    """
    try:
        named_variables = x.get_named_variables()
        trace = x.get_trace()

        val = trace['val']

        updated_trace = {}
        updated_trace.update(trace)

        updated_trace['val'] = operation(val)

        for var in named_variables:
            updated_trace[f'd_{var}'] = doperation(val) * updated_trace[f'd_{var}']

        return AutoDiff(name=named_variables,
                        trace=updated_trace)
    except:
        return operation(x)

def exp(x):
    return __update_unary(x, np.exp, np.exp)

def dlog(x):
    return 1./x
   
def log(x):
    return __update_unary(x, np.log, dlog)

def dsqrt(x):
    return 1/2 * 1/np.sqrt(x)

def sqrt(x):
    return __update_unary(x, np.sqrt, dsqrt)

def sin(x):
    return __update_unary(x, np.sin, np.cos)

def dcos(x):
    return -np.sin(x)

def cos(x):
    return __update_unary(x, np.cos, dcos)

def dtan(x):
    return 1./(np.cos(x)**2)

def tan(x):
    return __update_unary(x, np.tan, dtan)

def csc(x):
    return 1/sin(x)

def sec(x):
    return 1/cos(x)

def cot(x):
    return 1/tan(x)

def asin(x):
    return arcsin(x)

def darcsin(x):
    return 1/np.sqrt(1 - x**2)

def arcsin(x):
    return __update_unary(x, np.arcsin, darcsin)

def acos(x):
    return arccos(x)

def darccos(x):
    return -1./np.sqrt(1 - x**2)

def arccos(x):
    return __update_unary(x, np.arccos, darccos)
    
def atan(x):
    return arctan(x)

def darctan(x):
    return 1/(1 + x**2)

def arctan(x):
    return __update_unary(x, np.arctan, darctan)

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
    return __update_unary(x, np.sinh, np.cosh)

def cosh(x):
    return __update_unary(x, np.cosh, np.sinh)

def dtanh(x):
    return 1/np.cosh(x)**2

def tanh(x):
    return __update_unary(x, np.tanh, dtanh)

def csch(x):
    return 1/sinh(x)

def sech(x):
    return 1/cosh(x)

def coth(x):
    return 1/tanh(x)

def asinh(x):
    return arcsinh(x)

def darcsinh(x):
    return 1/np.sqrt(1 + x**2)

def arcsinh(x):
    return __update_unary(x, np.arcsinh, darcsinh)
    
def acosh(x):
    return arccosh(x)

def darccosh(x):
    return 1/np.sqrt(1 + x) * 1/np.sqrt(x - 1)

def arccosh(x):
    return __update_unary(x, np.arccosh, darccosh)  

def atanh(x):
    return arctanh(x)

def darctanh(x):
    return 1/(1 - x**2)

def arctanh(x):
    return __update_unary(x, np.arctanh, darctanh)  

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






